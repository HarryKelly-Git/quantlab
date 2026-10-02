"""Offline fixtures for the free alt-data providers (shapes as published by SEC EDGAR / the House Clerk)."""
from __future__ import annotations

import io
import zipfile

FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <schemaVersion>X0508</schemaVersion>
  <documentType>4</documentType>
  <periodOfReport>2024-03-01</periodOfReport>
  <issuer><issuerCik>0000111111</issuerCik><issuerName>Alpha Corp</issuerName><issuerTradingSymbol>aaa</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0000222222</rptOwnerCik><rptOwnerName>Doe Jane</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isDirector>1</isDirector><isOfficer>true</isOfficer><officerTitle>Chief Executive Officer</officerTitle>
      <isTenPercentOwner>0</isTenPercentOwner><isOther>0</isOther></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2024-03-01</value></transactionDate>
      <transactionCoding><transactionFormType>4</transactionFormType><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>10.00</value><footnoteId id="F1"/></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2024-03-01</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>500</value></transactionShares>
        <transactionPricePerShare><value>12.5</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2024-02-28</value></transactionDate>
      <transactionCoding><transactionCode>F</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>200</value></transactionShares>
        <transactionPricePerShare><value>11</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
  <derivativeTable>
    <derivativeTransaction>
      <securityTitle><value>Stock Option</value></securityTitle>
      <transactionDate><value>2024-02-28</value></transactionDate>
      <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>300</value></transactionShares>
        <transactionPricePerShare><value>0</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode></transactionAmounts>
    </derivativeTransaction>
  </derivativeTable>
</ownershipDocument>"""

FORM4_SUBMISSION = f"""<SEC-DOCUMENT>0000222222-24-000001.txt : 20240304
<SEC-HEADER>0000222222-24-000001.hdr.sgml : 20240304
<ACCEPTANCE-DATETIME>20240304173015
ACCESSION NUMBER:		0000222222-24-000001
CONFORMED SUBMISSION TYPE:	4
</SEC-HEADER>
<DOCUMENT>
<TYPE>4
<TEXT>
<XML>
{FORM4_XML}
</XML>
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>"""

FORM4_NO_TICKER = FORM4_XML.replace("<issuerTradingSymbol>aaa</issuerTradingSymbol>",
                                    "<issuerTradingSymbol>NONE</issuerTradingSymbol>")

FORM_INDEX = """Description:           Daily Index of EDGAR Dissemination Feed by Form Type
Last Data Received:    March 4, 2024
Comments:              webmaster@sec.gov

Form Type   Company Name                                                  CIK         Date Filed  File Name
---------------------------------------------------------------------------------------------------------------------------------------------
10-K        BIG CO                                                        999999      20240304    edgar/data/999999/0000999999-24-000009.txt
4           Alpha Corp                                                    111111      20240304    edgar/data/111111/0000222222-24-000001.txt
4           Doe Jane                                                      222222      20240304    edgar/data/222222/0000222222-24-000001.txt
4/A         Beta  Holdings Inc                                            333333      20240304    edgar/data/333333/0000333333-24-000002.txt
"""

CURRENT_ATOM = """<?xml version="1.0" encoding="ISO-8859-1" ?>
<feed xmlns="http://www.w3.org/2005/Atom">
<title>Latest Filings</title>
<entry>
<title>4 - Gamma Inc (0000444444) (Issuer)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/444444/000044444424000003/0000444444-24-000003-index.htm"/>
<updated>2024-03-05T09:15:00-05:00</updated>
<category scheme="https://www.sec.gov/" label="form type" term="4"/>
<id>urn:tag:sec.gov,2008:accession-number=0000444444-24-000003</id>
</entry>
<entry>
<title>4 - Smith John (0000555555) (Reporting)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/555555/000044444424000003/0000444444-24-000003-index.htm"/>
<updated>2024-03-05T09:15:00-05:00</updated>
<category scheme="https://www.sec.gov/" label="form type" term="4"/>
<id>urn:tag:sec.gov,2008:accession-number=0000444444-24-000003</id>
</entry>
<entry>
<title>8-K - Other Co (0000666666) (Filer)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/666666/000066666624000001/0000666666-24-000001-index.htm"/>
<updated>2024-03-05T09:10:00-05:00</updated>
<category scheme="https://www.sec.gov/" label="form type" term="8-K"/>
<id>urn:tag:sec.gov,2008:accession-number=0000666666-24-000001</id>
</entry>
</feed>"""

HOUSE_INDEX_XML = """<?xml version="1.0" encoding="utf-8"?>
<FinancialDisclosure>
  <Member><Prefix>Hon.</Prefix><Last>Example</Last><First>Pat</First><Suffix /><FilingType>P</FilingType>
    <StateDst>CA12</StateDst><Year>2024</Year><FilingDate>3/1/2024</FilingDate><DocID>20024001</DocID></Member>
  <Member><Prefix>Hon.</Prefix><Last>Paper</Last><First>Lee</First><Suffix /><FilingType>P</FilingType>
    <StateDst>TX07</StateDst><Year>2024</Year><FilingDate>3/4/2024</FilingDate><DocID>8220999</DocID></Member>
  <Member><Prefix /><Last>Annual</Last><First>Sam</First><Suffix /><FilingType>O</FilingType>
    <StateDst>NY03</StateDst><Year>2024</Year><FilingDate>3/2/2024</FilingDate><DocID>10055555</DocID></Member>
</FinancialDisclosure>"""


def house_index_zip(xml: str = HOUSE_INDEX_XML) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("2024FD.txt", "Prefix\tLast\tFirst\n")
        z.writestr("2024FD.xml", xml)
    return buf.getvalue()


# text layer of an electronically filed PTR, as a PDF text extractor returns it (wrapped cells,
# labelled sub-lines, a NUL glyph artefact)
PTR_TEXT = """Clerk of the House of Representatives • Legislative Resource Center
P T R
Name: Hon. Pat Example
Status: Member
State/District: CA12
T
ID Owner Asset Transaction
Type
Date Notification
Date
Amount Cap.
Gains >
$200?
SP Alpha Corp - Common Stock (AAA) [ST] P 02/14/2024 02/20/2024 $1,001 -
$15,000
F\x00 S\x00 : New
D : Bought after the earnings call; not a description that should leak (BBB) [ST] P 01/01/2024
Beta Holdings Inc. Class A
(BBB) [ST]
S (partial) 02/01/2024 02/02/2024 $50,001 - $100,000
F S : New
JT Gamma Inc (CCC) [OP] P 02/05/2024 02/06/2024 $15,001 - $50,000
D : Purchased 10 call options
US Treasury Bill [GS] P 02/07/2024 02/08/2024 Over $50,000,000
F S : New
"""
