import base64
import textwrap

import pandas as pd
import pytest

from octomatrix.data import build
from octomatrix.data.http_parse import clean_headers, parse_raw_request, split_uri
from octomatrix.data.loaders import access_log, cicids2018, crs_regression, csic2010, ecml_pkdd2007
from octomatrix.data.schema import to_frame, validate

CSIC_ATTACK = textwrap.dedent(
    """\
    Start - Id: 7
    class: Attack
    POST http://localhost:8080/tienda1/publico/anadir.jsp HTTP/1.1
    User-Agent: Mozilla/5.0 (compatible; Konqueror/3.5; Linux)
    Host: localhost:8080
    Cookie: JSESSIONID=AE29AEEBDE479D5E1A18B4108C8E3CE0
    Content-Type: application/x-www-form-urlencoded

    id=2&cantidad=%27%3B+DROP+TABLE+usuarios

    End - Id: 7
    """
)

CSIC_VALID = textwrap.dedent(
    """\
    Start - Id: 1
    class: Valid
    GET http://localhost:8080/tienda1/index.jsp?a=1 HTTP/1.1
    Host: localhost:8080
    Cookie: JSESSIONID=81761ACA043B0E6014CA42A4BCD06AB5

    null

    End - Id: 1
    """
)

ECML = textwrap.dedent(
    """\
    Start - Id: 5
    class: OsCommanding
    GET /scripts/file.bat" & dir c:/ .exe? HTTP/1.0
    Host: www.ownti.fr
    Accept: */*

    null

    End - Id: 5
    Start - Id: 6
    class: PathTransversal
    GET /a.gif?x=1 HTTP/1.1
    Host: www.x.cz
    Cookie: 6hAd=../../../WINDOWS/autoexec.bat
    ----: -------------

    null

    End - Id: 6
    Start - Id: 8
    class: Valid
    GET /b.gif?y=2 HTTP/1.1
    Host: www.y.cz

    null

    End - Id: 8
    """
)


def test_split_uri_absolute_and_relative():
    assert split_uri("http://localhost:8080/a/b.jsp?x=1&y=2") == ("/a/b.jsp", "x=1&y=2")
    assert split_uri("/post?q=1") == ("/post", "q=1")
    assert split_uri("") == ("/", "")


def test_clean_headers_drops_noise_and_session_ids():
    out = clean_headers(
        [("Host", "x"), ("Cookie", "JSESSIONID=ABC; pref=1"), ("User-Agent", "sqlmap"), ("----", "---")]
    )
    assert out == "Cookie: pref=1\nUser-Agent: sqlmap"


def test_parse_raw_request_with_spaces_in_uri():
    req = parse_raw_request('GET /cgi-bin/x.exe 127.0.0.1? HTTP/1.0\nHost: h\n\nnull')
    assert req["path"] == "/cgi-bin/x.exe 127.0.0.1"
    assert req["body"] == ""


def test_csic_loader(tmp_path):
    (tmp_path / "cisc_anomalousTraffic_test.txt").write_text(CSIC_ATTACK)
    (tmp_path / "cisc_normalTraffic_train.txt").write_text(CSIC_VALID)
    df = to_frame(csic2010.load(tmp_path))
    attack = df[df.label == 1].iloc[0]
    assert attack.method == "POST"
    assert attack.path == "/tienda1/publico/anadir.jsp"
    assert "DROP+TABLE" in attack.body
    assert attack.attack_type == "unspecified"
    assert attack.origin_split == "test"
    assert "JSESSIONID" not in attack.headers
    normal = df[df.label == 0].iloc[0]
    assert (normal.query, normal.body, normal.origin_split) == ("a=1", "", "train")


def test_ecml_loader_keeps_attacks_with_spaces_and_header_payloads(tmp_path):
    (tmp_path / "xml_train.txt").write_text("")
    (tmp_path / "xml_test.txt").write_text(ECML)
    df = to_frame(ecml_pkdd2007.load(tmp_path))
    assert list(df.attack_type) == ["command_injection", "path_traversal", "normal"]
    assert '& dir c:/' in df.iloc[0].path
    assert "autoexec.bat" in df.iloc[1].headers
    assert "----" not in df.iloc[1].headers


def test_ecml_unknown_class_fails(tmp_path):
    (tmp_path / "xml_train.txt").write_text(ECML.replace("class: Valid", "class: Weird"))
    (tmp_path / "xml_test.txt").write_text("")
    with pytest.raises(ValueError, match="Weird"):
        list(ecml_pkdd2007.load(tmp_path))


def test_crs_loader(tmp_path):
    sqli = tmp_path / "REQUEST-942-APPLICATION-ATTACK-SQLI"
    proto = tmp_path / "REQUEST-920-PROTOCOL-ENFORCEMENT"
    sqli.mkdir()
    proto.mkdir()
    raw = base64.b64encode(b"GET /?id=1%20union%20select HTTP/1.1\r\nHost: localhost\r\n\r\n").decode()
    (sqli / "942100.yaml").write_text(
        textwrap.dedent(
            f"""\
            rule_id: 942100
            tests:
              - test_id: 1
                stages:
                  - input:
                      method: POST
                      uri: "/post"
                      headers: {{Host: localhost, User-Agent: "OWASP CRS test agent"}}
                      data: "var=1234 OR 1=1"
                    output:
                      log: {{expect_ids: [942100]}}
              - test_id: 2
                stages:
                  - input:
                      uri: "/get?q=hello world"
                    output:
                      log: {{no_expect_ids: [942100]}}
              - test_id: 3
                stages:
                  - input:
                      encoded_request: "{raw}"
                    output:
                      log: {{expect_ids: [942100]}}
              - test_id: 4
                stages:
                  - input: {{uri: "/x"}}
                    output: {{status: 200}}
            """
        )
    )
    (proto / "920100.yaml").write_text("rule_id: 920100\ntests:\n  - test_id: 1\n    stages:\n      - input: {uri: /}\n        output: {log: {expect_ids: [920100]}}\n")
    df = to_frame(crs_regression.load(tmp_path))
    assert len(df) == 3  # sin el test de status ni la familia 920
    assert list(df.source) == ["crs_regression", "crs_regression_negative", "crs_regression"]
    assert list(df.attack_type) == ["sqli", "normal", "sqli"]
    assert df.iloc[0].body == "var=1234 OR 1=1"
    assert df.iloc[1].query == "q=hello world"
    assert df.iloc[2].query == "id=1%20union%20select"


def test_access_log_loader(tmp_path):
    log = tmp_path / "app.log"
    log.write_text(
        '1.2.3.4 - - [03/Oct/2025:17:47:05 +0000] "GET /eventos?cat=rock HTTP/1.1" 200 512 "-" "Mozilla/5.0"\n'
        '1.2.3.4 - - [03/Oct/2025:17:47:06 +0000] "GET /wp-admin HTTP/1.1" 404 0 "-" "nikto"\n'
        '127.0.0.1 - - [03/Oct/2025 17:47:05] "GET / HTTP/1.1" 200 -\n'
        "basura\n"
    )
    df = to_frame(access_log.load(log, source="pereira"))
    assert list(df.path) == ["/eventos", "/"]
    assert df.iloc[0].query == "cat=rock"
    assert df.iloc[0].headers == "User-Agent: Mozilla/5.0"
    assert set(df.source) == {"own:pereira"}
    assert len(to_frame(access_log.load(log, source="p", drop_error_status=False))) == 3


def _row(**kw):
    base = dict(method="GET", path="/", query="", body="", headers="", label=0, attack_type="normal",
                source="s", origin_split="-", sample_id="")
    base.update(kw)
    return base


def test_deduplicate_drops_duplicates_and_conflicts():
    df = validate(pd.DataFrame([
        _row(query="a=1", sample_id="1"),
        _row(query="a=1", sample_id="2"),                                 # duplicado
        _row(query="b=1", sample_id="3"),
        _row(query="b=1", label=1, attack_type="sqli", sample_id="4"),   # conflicto con 3
        _row(query="c=1", label=1, attack_type="xss", sample_id="5"),
    ]))
    out, stats = build.deduplicate(df)
    assert list(out.sample_id) == ["1", "5"]
    assert stats["duplicate_rows_dropped"] == {"s": 1}
    assert stats["conflicting_rows_dropped"] == {"s": 2}


def test_validate_rejects_incoherent_labels():
    with pytest.raises(ValueError, match="incoherentes"):
        validate(pd.DataFrame([_row(label=1)]))
    with pytest.raises(ValueError, match="attack_type desconocido"):
        validate(pd.DataFrame([_row(label=1, attack_type="ddos")]))


def test_cicids_cleaning(tmp_path):
    header = "Flow ID,Src IP,Src Port,Dst IP,Dst Port,Protocol,Timestamp,Flow Duration,Flow Byts/s,Label"
    csv = tmp_path / "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
    csv.write_text(
        "\n".join([
            header,
            "x,10.0.0.1,1,10.0.0.2,80,6,20/02/2018 08:00:00,100,Infinity,Benign",
            header,  # cabecera repetida a mitad de archivo
            "y,10.0.0.3,2,10.0.0.2,80,6,20/02/2018 08:00:01,50,12.5,Brute Force -XSS",
        ]) + "\n"
    )
    df = pd.concat(cicids2018.load_day(csv))
    assert len(df) == 2
    assert not {"Flow ID", "Src IP", "Src Port", "Dst IP"} & set(df.columns)
    assert list(df.attack_type) == ["normal", "brute_force_xss"]
    assert list(df.label) == [0, 1]
    assert pd.isna(df["Flow Byts/s"].iloc[0])
    assert df["timestamp"].notna().all()
