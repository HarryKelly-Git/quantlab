# External services and research methods: verified facts

Produced on 2026-09-24 by a research pass against primary documentation (docs.alpaca.markets, sec.gov,
provider API docs, original papers). `verified=true` means the fact was read on a primary page or confirmed
by a live request in that session. Anything unverified is listed under UNKNOWN and must be treated as UNKNOWN
in code (fail safe; never assume). Re-verify before relying on any fact older than a few months.

## Alpaca Market Data API

_Alpaca Market Data API (docs.alpaca.markets), free Basic plan: historical stock bars, rate limits, corporate actions, news, delisted and renamed symbols, and market calendar/clock. Checked 2026-09-24 against the docs' .md/OpenAPI sources, plus unauthenticated live probes._

1. **Market data base URL is https://data.alpaca.markets (the sandbox host is for broker partners only). Trading, calendar, clock and assets use a different host.** [verified]  
   Historical base: https://data.alpaca.markets/{version}. Broker sandbox: https://data.sandbox.alpaca.markets. Trading API: https://paper-api.alpaca.markets (paper) or https://api.alpaca.markets (live). Trading API auth uses headers APCA-API-KEY-ID and APCA-API-SECRET-KEY. Market data keys are tied to owner_id, so paper and live keys see the same data subscription (per an Alpaca staff forum post). Live probe on 2026-09-24 with no key: every data.alpaca.markets and paper-api path returned HTTP 401, text/html on the data host. A made-up path (/v9/nonexistent) also returned 401, so a 401 does not prove an endpoint exists. The docs FAQ says unauthenticated market data requests get 403, but the live probe got 401.  
   Source: https://docs.alpaca.markets/us/docs/historical-api

2. **The multi-symbol historical bars endpoint is GET /v2/stocks/bars.** [verified]  
   Query params: symbols (required, comma-separated), timeframe (required), start, end, limit, adjustment, asof, feed, currency (ISO 4217, default USD), page_token, sort. Response shape: {"bars": {"AAPL": [bar,...], ...}, "next_page_token": string|null, "currency"?: string}. bars and next_page_token are required keys. operationId StockBars.  
   Source: https://docs.alpaca.markets/us/reference/stockbars

3. **The single-symbol bars endpoint is GET /v2/stocks/{symbol}/bars. It takes the same parameters and returns a flat array.** [verified]  
   Response: {"bars": [bar,...], "symbol": "AAPL", "next_page_token": string|null, "currency"?: string}. bars, next_page_token and symbol are required. operationId StockBarSingle.  
   Source: https://docs.alpaca.markets/us/reference/stockbarsingle-1

4. **Allowed timeframe values.** [verified]  
   [1-59]Min or [1-59]T; [1-23]Hour or [1-23]H; 1Day or 1D; 1Week or 1W; [1,2,3,4,6,12]Month or [1,2,3,4,6,12]M. Example: timeframe=1Day.  
   Source: https://docs.alpaca.markets/us/reference/stockbars

5. **start and end are both inclusive and accept RFC-3339 or YYYY-MM-DD. Their defaults depend on real-time entitlement.** [verified]  
   Examples: 2024-01-04T01:02:03.123456789Z, 2024-01-04T00:00:00Z, 2024-01-03T09:30:00-04:00, 2024-01-03. Default start: beginning of the current day, but at least 15 minutes ago if the user has no real-time access for the feed. Default end: now with real-time access for the feed, otherwise now minus 15 minutes. A 2024 staff forum post says end defaulted to 'now' and caused SIP errors, which conflicts with the current docs, so always pass end explicitly.  
   Source: https://docs.alpaca.markets/us/reference/stockbars

6. **limit defaults to 1000 and maxes at 10000. It caps the total bars per page across all symbols, not per symbol. A page can be short even when more data exists.** [verified]  
   From the reference: "The API may return less, even if there are more available data points... Always check the next_page_token". Alpaca staff (forum, 2025-04) say short pages happen when the backend is waiting on the database. Loop until next_page_token is null. Never treat len(page) < limit as the end.  
   Source: https://docs.alpaca.markets/us/reference/stockbars

7. **Multi-symbol results are ordered by symbol first, then by timestamp. sort is asc (default) or desc.** [verified]  
   The first page may contain only the alphabetically first symbol. Keep paging with next_page_token until you reach the other symbols. page_token is opaque. The documented example tokens decode from base64 to 'AAPL|D|2023-09-29T04:00:00.000000000Z', i.e. symbol|timeframe code|timestamp of the last returned bar. Pass the same other query params when using the token.  
   Source: https://docs.alpaca.markets/us/reference/stockbars

8. **The adjustment parameter defaults to raw (no adjustment). Values can be combined with commas.** [verified]  
   raw = no adjustment. split = price and volume adjusted for forward and reverse splits. dividend = price adjusted for cash dividends. spin-off = price adjusted for spin-offs (hyphen here; the corporate-actions types use spin_off with an underscore). all = every adjustment above. Example: adjustment=split,spin-off.  
   Source: https://docs.alpaca.markets/us/reference/stockbars

9. **The feed enum is iex | otc | sip | boats. Docs disagree on the default, so always pass feed explicitly.** [verified]  
   The OpenAPI schema says default "sip". The Market Data FAQ says "The default value for feed is always the 'best' available feed based on the user's subscription", and gives iex as the default for a no-subscription latest-trade call. Alpaca staff recommend always setting feed explicitly. The historical-stock-data guide also describes an 'overnight' feed (Alpaca's 15-minute-delayed version of BOATS) that is not in the bars enum. Only iex works without a subscription per that guide, yet sip history is still allowed when end is at least 15 minutes old (next fact).  
   Source: https://docs.alpaca.markets/us/docs/market-data-faq

10. **On the free Basic plan, historical SIP data can be queried as long as end is at least 15 minutes old. Newer SIP data is rejected.** [verified]  
   The plans table lists the Basic historical limitation as 'latest 15 minutes' (Algo Trader Plus: no restriction). FAQ: "For historical queries, the end parameter must be at least 15 minutes old to query SIP data without a subscription." Error body: {"code":42210000,"message":"subscription does not permit querying recent SIP data"}, which the FAQ covers in its 403 section. An Alpaca staff forum post says the older SIP history is identical to the paid plan's.  
   Source: https://docs.alpaca.markets/us/docs/about-market-data-api

11. **The Paper Trading doc says Paper Only accounts get IEX data only, which conflicts with Basic-plan SIP history access.** [verified]  
   Quote: "As an Alpaca Paper Only Account holder, you are only entitled to receive and make use of IEX market data." The same page says paper trading does NOT simulate dividends. Paper Only accounts are the kind open to anyone globally.  
   Source: https://docs.alpaca.markets/us/docs/paper-trading

12. **OTC data (feed=otc) needs a special subscription that is currently offered only to broker partners.** [verified]  
   FAQ: "Market data for OTC symbols can only be queried with a special subscription currently available only for broker partners." Use GET https://api.alpaca.markets/v2/assets/{symbol}: exchange == "OTC" means no data on Basic.  
   Source: https://docs.alpaca.markets/us/docs/market-data-faq

13. **History depth: the plans table says 'Since 2016' for both plans. Per Alpaca staff, IEX bars go back only to 2020.** [verified]  
   Plans table row: 'Historical data timeframe: Since 2016'. Alpaca Developer Relations (forum, 2023-05-23): "IEX bar data is only available from 2020. However, sip data is available from 2016."  
   Source: https://docs.alpaca.markets/us/docs/about-market-data-api

14. **Rate limit for historical market data on the free Basic plan is 200 calls per minute (Algo Trader Plus: 10,000 per minute). Rate-limit headers are returned.** [verified]  
   Headers: X-RateLimit-Limit (per-minute limit), X-RateLimit-Remaining, and X-RateLimit-Reset (UNIX epoch when the remaining quota changes). HTTP 429 means you hit the limit. Basic plan websocket limit: 30 symbols. Options on Basic: also 200 per minute.  
   Source: https://docs.alpaca.markets/us/docs/about-market-data-api

15. **The Trading API (orders, account, assets, clock, calendar) is throttled at 200 requests per minute per account and returns 429 when exceeded.** [verified]  
   Alpaca support article (dated December 2022): "currently 200 requests per minute, per account... a '429- Too Many Requests' status will be returned". Users cannot raise the limit themselves.  
   Source: https://alpaca.markets/support/usage-limit-api-calls

16. **Bar field names. All eight fields are required.** [verified]  
   t = timestamp (RFC-3339, nanosecond precision), o = open, h = high, l = low, c = close (all double), v = volume (int64), n = trade count (int64), vw = VWAP (double). Response keys: bars, next_page_token, symbol (single-symbol endpoint only), currency.  
   Source: https://docs.alpaca.markets/us/reference/stockbars

17. **A daily bar's t is midnight America/New_York for that trading day, written in UTC: T04:00:00Z during daylight time and T05:00:00Z in winter. t is the start of the interval, not when the data became available.** [verified]  
   FAQ: trade SIP timestamps are "truncated... to the day (in New York) for daily bars", and "The timestamp of the bar is the left side of the interval." Example: the AAPL 2023-09-29 daily bar has t=2023-09-29T04:00:00Z but includes the 16:00 ET close. The earliest you can know that close is about 16:00 ET that day, so treating t as the availability time leaks future data.  
   Source: https://docs.alpaca.markets/us/docs/market-data-faq

18. **Daily bar aggregation rules: extended-hours and odd-lot trades add to volume but do not set prices. A bar is emitted only if none of o, h, l, c, v is 0, so days with no qualifying trades are missing rather than zero-filled.** [verified]  
   Daily bars: condition T (Extended Hours) and U update volume only. I (odd lot) and C, N, R, V, 7, B (on tapes A/B), H and W also update volume only. M and Q (official close/open reports) update nothing. 9 (Corrected Consolidated Close) updates the daily open/close and high/low but not volume. When a trade has several conditions, the strictest rule applies. vw uses a separate volume total that counts only trades updating both high/low and volume, so vw*v is not necessarily the dollar volume. Hour, week and month bars are built from minute or daily bars: first open, max high, min low, last close, summed volume and n, volume-weighted VWAP.  
   Source: https://docs.alpaca.markets/us/docs/market-data-faq

19. **The asof parameter (YYYY-MM-DD, default today) links an entity's data across symbol renames. It is on by default. asof=- turns the mapping off.** [verified]  
   The reference says asof identifies the underlying entity of the queried symbol on that date, and data under past symbols is returned if the range spans the rename. Past-symbol data comes back labelled with the queried symbol. If the symbol is not found on the asof date, no mapping happens and data comes back by raw symbol. Example: FB became META on 2022-06-09. Querying META (default asof) returns Mon–Wed 2022-06-06..08 bars that traded as FB, all under key META. With asof=- you get only bars from 2022-06-09 on. Querying FB with asof=2022-06-06 returns the whole week under key FB. Querying FB with a post-rename asof returns only FB-ticker data.  
   Source: https://docs.alpaca.markets/us/docs/market-data-faq

20. **Rename mapping only takes effect the day after a rename.** [verified]  
   FAQ: "the asof mapping is only available on our historical endpoints the day after the rename". For FB to META it was available from 2022-06-10, and queries on 2022-06-09 did not return FB bars. Latest-trade, latest-quote, latest-bar and snapshot endpoints are never adjusted or mapped (there is no adjustment parameter on latest bars). Streams use whatever symbol is current, so clients must resubscribe after a rename.  
   Source: https://docs.alpaca.markets/us/docs/market-data-faq

21. **Bars for delisted symbols can still be queried by their old ticker. However, the Trading API's /v2/assets lists only assets Alpaca currently holds, and there is no historical assets endpoint, so you cannot list a survivorship-free universe from Alpaca.** [UNVERIFIED]  
   Staff forum posts: 2022-03-04 (Gergely_Alpaca): "the delisted symbols remain in the database". 2025-12-18: "at the moment we do not have an endpoint for historical assets" (maybe future, possibly with asof). User report (2022): /v2/stocks/WFM/bars returned bars ending 2017-08-25 while /v2/assets did not include WFM. Delisted entries in the asset table may have their symbol replaced by a number or a suffix (Dan_Whitnable_Alpaca, staff). Sourced from Alpaca community forum staff posts, not reference docs.  
   Source: https://forum.alpaca.markets/t/delisted-tickers/18227

22. **GET /v2/assets on the Trading host is the current master asset list. Its status filter includes all statuses by default.** [verified]  
   Host: paper-api.alpaca.markets or api.alpaca.markets. Params: status (e.g. active; all statuses by default), asset_class (default us_equity), exchange (AMEX, ARCA, BATS, NYSE, NASDAQ, NYSEARCA, OTC, CRYPTO), attributes (comma list: ptp_no_exception, ptp_with_exception, ipo, has_options, options_late_close, fractional_eh_enabled, overnight_tradable, overnight_halted). Fields include id, class, exchange, symbol, name, status, tradable, cusip, marginable, shortable, fractionable, easy_to_borrow, borrow_status. There is no listing date and no delisting date.  
   Source: https://docs.alpaca.markets/us/reference/get-v2-assets-1

23. **The corporate actions endpoint is GET /v1/corporate-actions on the data host. The old Trading API /v2/corporate_actions/announcements is deprecated.** [verified]  
   Params: symbols (comma list), cusips, types, region (us default, non_us, all), start and end (YYYY-MM-DD, inclusive, both default to the current day), ids (comma list of UUIDs; cannot be combined with other filters), limit (default 100, max 1000, counted across all symbols), data_quality (complete default, all), page_token, sort (asc default, desc). The start/end descriptions say results are "sorted by their process_date". Rate-limit headers are returned.  
   Source: https://docs.alpaca.markets/us/reference/corporateactions-1

24. **The corporate action types (16 of them) and the response keys they come back under.** [verified]  
   types= reverse_split, forward_split, unit_split, cash_dividend, stock_dividend, spin_off, cash_merger, stock_merger, stock_and_cash_merger, redemption, name_change, worthless_removal, rights_distribution, partial_call, reorganization, capital_gains_distribution. Response: {"corporate_actions": {"forward_splits": [...], "reverse_splits": [...], "unit_splits", "cash_dividends", "stock_dividends", "spin_offs", "cash_mergers", "stock_mergers", "stock_and_cash_mergers", "redemptions", "name_changes", "worthless_removals", "rights_distributions", "partial_calls", "reorganizations", "capital_gains_distributions"}, "next_page_token": ...}. Each record has id (UUID).  
   Source: https://docs.alpaca.markets/us/reference/corporateactions-1

25. **Date fields on corporate actions differ by type. No type has a declaration or announcement date, or a record creation timestamp.** [verified]  
   Definitions: process_date = when Alpaca processes the action. ex_date = cutoff for shareholders to be credited. record_date = when you must own shares. payable_date = when the benefit is paid. effective_date is used for mergers. due_bill_redemption_date also exists. cash_dividend: id, symbol, cusip, rate (double), special, foreign, process_date, ex_date (required); record_date, payable_date, due_bill_on_date, due_bill_off_date, sub_type (interest|return_of_capital), currency, isin (optional). forward_split: old_rate, new_rate, ex_date, record_date, payable_date, due_bill_redemption_date. reverse_split: old_rate, new_rate, old_cusip, new_cusip, new_symbol (empty if the ticker is unchanged). spin_off: source_symbol, source_rate, new_symbol, new_rate, ex_date. name_change: old_symbol, new_symbol, old_cusip, new_cusip, process_date only (no ex_date). cash_merger: acquiree_symbol, rate, effective_date. stock_merger: acquirer_symbol, acquirer_rate, acquiree_symbol, acquiree_rate, effective_date. worthless_removal: symbol, cusip, process_date.  
   Source: https://docs.alpaca.markets/us/reference/corporateactions-1

26. **By default, incomplete corporate actions are filtered out, and Alpaca does not guarantee when records appear.** [verified]  
   data_quality=complete (default) leaves out actions that are missing required fields (e.g. ex-date or CUSIP/ISIN) and are not yet processed; processed ones always appear. data_quality=all returns early, incomplete records. Docs warning: "Currently Alpaca has no guarantees on the creation time of corporate actions... may not be available immediately after they are announced."  
   Source: https://docs.alpaca.markets/us/reference/corporateactions-1

27. **Corporate action records can be updated or deleted after they are published. A server-sent events (SSE) stream carries those changes with timestamps and can replay history.** [verified]  
   GET https://data.alpaca.markets/v1beta1/events/corporate-actions (text/event-stream). Params: type (comma list of <type>_corporateaction_event), region (all default|us|non_us), since (RFC-3339; required if until is set), until (inclusive; cannot be in the future), since_id / until_id (event ULIDs). The Last-Event-Id header resumes inclusively, so deduplicate by event_id. Event envelope: event_id (26-char ULID, increasing), at (when the event was emitted), action (insert | update = e.g. date or rate correction | delete; ca.id matches the original), event_type, region, ca. Decimal fields such as rate are JSON strings on SSE but numbers on REST.  
   Source: https://docs.alpaca.markets/us/reference/subscribetocorporateactionseventssse

28. **The deprecated Trading API announcements endpoint limited each query to a 90-day window.** [verified]  
   GET /v2/corporate_actions/announcements (trading host, deprecated). Required: ca_types, since, until (YYYY-MM-DD). Optional: symbol, cusip, date_type (declaration_date|ex_date|record_date|payable_date). "The date range is limited to 90 days." The docs no longer state any range limit for /v1/corporate-actions.  
   Source: https://docs.alpaca.markets/us/reference/get-v2-corporate_actions-announcements-1

29. **Adjusted bars use Alpaca's current corporate-action database, which has been corrected after the fact, so adjusted history can change between downloads.** [UNVERIFIED]  
   Forum 2024-10-10 (Gergely_Alpaca): an incorrect FI split "in our database" was removed, and a missing 2021-11-02 DELL 1973:1000 split was added. After each fix, adjustment=all bars for those periods changed. No adjustment 'as-of date' parameter exists.  
   Source: https://forum.alpaca.markets/t/adjusted-historic-bar-data-is-inaccurate/15101

30. **Bars can be revised after they are first calculated because of late or corrected trades.** [verified]  
   Alpaca Developer Relations (staff, forum 2024-05-29): bars are first calculated about 1 second after the bar closes, and typically fewer than 0.5% of trades get updated or added to a bar later. Extended-hours historical bars can differ from the live ones.  
   Source: https://forum.alpaca.markets/t/cant-read-spy-minutes/14293

31. **The news endpoint is GET /v1beta1/news on the data host.** [verified]  
   Params: start and end (RFC-3339 or YYYY-MM-DD, inclusive), sort (asc|desc, default desc, described as "Sort articles by updated date"), symbols (comma list; stocks and crypto, e.g. AAPL,TSLA,BTCUSD), limit (1–50; default 10 per the endpoint description), include_content (bool), exclude_contentless (bool), page_token. Response: {"news": [...], "next_page_token": string|null}.  
   Source: https://docs.alpaca.markets/us/reference/news-3

32. **News article fields.** [verified]  
   Required: id (int64), headline, author, created_at (RFC-3339), updated_at (RFC-3339), summary (may be the first sentence of content), content (may contain HTML), images (array of {size: thumb|small|large, url}), symbols (related or mentioned symbols), source (e.g. benzinga). Optional: url (string|null). Example: created_at 2021-12-31T11:08:42Z, updated_at 2021-12-31T11:08:43Z. Real-time stream: wss://stream.data.alpaca.markets/v1beta1/news, same fields plus T="n".  
   Source: https://docs.alpaca.markets/us/reference/news-3

33. **News pagination and sorting use updated_at, not created_at.** [verified]  
   The sort param is documented as 'Sort articles by updated date'. The documented next_page_token 'MTY0MDk0ODkyMzAwMDAwMDAwMHwyNDg0MzE3MQ==' decodes to '1640948923000000000|24843171', i.e. nanosecond epoch 2021-12-31T11:08:43Z plus id. That matches the example article's updated_at (11:08:43), not its created_at (11:08:42).  
   Source: https://docs.alpaca.markets/us/reference/news-3

34. **News history goes back to 2015 and comes from a single source, Benzinga, averaging 130+ articles per day.** [verified]  
   From the Historical News Data doc. A 2022 launch blog said news calls share the market-data plan rate limits (200/min free). A staff forum post (2024-06-25) says historical news should exactly match streamed news, but a user reported streamed articles missing from the history.  
   Source: https://docs.alpaca.markets/us/docs/historical-news-data

35. **The legacy US market calendar is GET /v2/calendar on the Trading host and covers 1970 to 2029.** [verified]  
   Hosts: https://paper-api.alpaca.markets or https://api.alpaca.markets. Params: start, end (inclusive), date_type (TRADING default | SETTLEMENT). Response is an array of {date: YYYY-MM-DD, open: 'HH:MM', close: 'HH:MM', session_open: 'HHMM' (e.g. 0400), session_close: 'HHMM' (e.g. 2000), settlement_date}. Early closes are reflected in open/close.  
   Source: https://docs.alpaca.markets/us/reference/legacycalendar

36. **Newer calendar and clock endpoints, /v3/calendar/{market} and /v3/clock, are also on the Trading host and return timestamps with explicit time zones.** [verified]  
   GET /v3/calendar/{market} (e.g. NYSE, XNYS, NASDAQ, XNAS, IEX, BOATS, OPRA...). Params: start (date, default today), end (date, default start + 1 week), timezone (enum: UTC; default is the market's zone). Response: {market: {acronym, name, timezone, mic, bic}, calendar: [{date, core_start, core_end, pre_start, pre_end, post_start, post_end, lunch_start, lunch_end, settlement_date}]} with RFC-3339 times, e.g. 2025-01-02T09:30:00-05:00. GET /v3/clock?markets=NYSE,...&time=<RFC-3339> returns {clocks: [{market, timestamp, is_market_day, next_market_open, next_market_close, phase: closed|pre|core|lunch|post, phase_until}]}.  
   Source: https://docs.alpaca.markets/us/reference/calendar-2

37. **The legacy clock is GET /v2/clock on the Trading host.** [verified]  
   Response: {timestamp, is_open (bool), next_open, next_close}, as RFC-3339 with ET offset, e.g. 2025-06-24T16:00:00-04:00.  
   Source: https://docs.alpaca.markets/us/reference/legacyclock

38. **When one ticker is reused by different companies over time, querying it by raw symbol can return one series that mixes both companies' histories.** [UNVERIFIED]  
   User report (forum 2022-01, IEX feed): HLTH returned Nobilis Health bars (delisted 2019), then filler bars, then Cue Health bars from its 2021-09-24 IPO, in one series. Another user said it was an OTC uplisting. This predates the current asof and bar-emission rules, so current behavior is unconfirmed.  
   Source: https://forum.alpaca.markets/t/get-bars-returning-data-from-2-different-companies/7975

### Gotchas

- Daily bar t (e.g. 2023-09-29T04:00:00Z) is midnight New York time at the start of the session, but the bar holds that day's 16:00 ET close. Stamping features or signals with t leaks one session of future data. The bar becomes known at the earliest at 16:00 ET, and is final some time later because late or corrected trades can still revise it.
- t switches between T04:00:00Z (daylight time) and T05:00:00Z (winter). Converting to a UTC date is fine, but joining on the exact UTC timestamp across the daylight-saving change will misalign. Key daily bars on the New York calendar date instead.
- adjustment defaults to raw. Returns computed from raw bars show fake jumps at splits and ex-dividend dates. The opposite trap: adjustment=all/split/dividend is recomputed from Alpaca's current corporate-action database, which has been corrected after the fact (FI and DELL splits in 2024), so adjusted history is NOT point-in-time and can change between downloads. Store raw bars and apply your own versioned adjustments.
- Silent truncation: limit counts all symbols together (max 10000), results are ordered by symbol then time, and a page can be short even when more data exists. Stopping when len < limit, or not following next_page_token until null, quietly drops later symbols or dates.
- The default feed is ambiguous (the schema says sip, the FAQ says best available for the subscription). An omitted feed can silently switch between IEX (about 2.5% of volume, history only from 2020) and SIP, changing v, n, vw and even prices. Always send feed=sip or feed=iex and record it with the data.
- On the free plan, a feed=sip request whose end is within the last 15 minutes fails with code 42210000 'subscription does not permit querying recent SIP data'. Always set end to at least now minus 15 minutes, or cap at the prior close, and never rely on the default end.
- The Paper Trading doc says Paper Only accounts are entitled to IEX data only. If the operator's account is Paper Only (open to anyone globally, e.g. from NZ), relying on feed=sip history may break Alpaca's terms or fail even though the free plan otherwise allows SIP data older than 15 minutes.
- Survivorship bias: /v2/assets lists only assets Alpaca currently holds (delisted ones may be renamed with numbers or suffixes), and there is no historical-assets endpoint. Building a backtest universe from /v2/assets drops delisted, acquired and bankrupt names. Bars for delisted tickers reportedly still exist (per staff), but you must get the old tickers from an outside source.
- asof is on by default (asof = today). Querying today's ticker quietly merges the entity's history under old tickers and labels it with the new one. Querying an old ticker with default asof returns only that ticker's own data. Mapping only appears the day after a rename, so a same-day rerun gives different results. For reproducible research, pin asof explicitly (or use asof=-) and store it.
- Ticker reuse: a raw-symbol query (asof=- or symbol not found on the asof date) can splice two unrelated companies that used the same ticker into one series (HLTH forum case). Check for gaps and price discontinuities around IPO or delisting dates.
- Bars are emitted only when o, h, l, c and v are all non-zero. Days with no qualifying trades are simply missing (no zero-volume filler), so forward-fill or calendar-align explicitly using /v2/calendar or /v3/calendar/XNYS.
- Daily v and n include extended-hours and odd-lot volume, but daily o, h, l, c exclude extended-hours prices. vw's volume base differs from v, so vw*v is not dollar volume.
- Corporate actions have no declaration or announcement date and no created_at, and Alpaca does not guarantee when records appear. Records can later be updated (date or rate fixes) or deleted. Querying /v1/corporate-actions for a past range returns today's corrected view, not what was knowable then. Only the SSE stream's `at` emission timestamps give point-in-time knowledge, and only from when you start recording or from however far back replay goes (unknown).
- /v1/corporate-actions start and end both default to TODAY. Omitting them silently returns only today's actions. data_quality=complete (default) also hides incomplete, unprocessed announcements, i.e. the most recent ones.
- Naming mismatch: the bars adjustment value is 'spin-off' (hyphen), the corporate-action types filter uses 'spin_off' (underscore), and the response keys are plural ('spin_offs', 'cash_dividends', ...). name_change records have only process_date (no ex_date), and mergers use effective_date.
- REST corporate-action rates are JSON numbers (double), while the SSE stream sends decimal strings ('0.24'). Parse both into Decimal to avoid float drift and type mismatches.
- News sort and pagination use updated_at, not created_at. An article edited after publication is ordered by its edit time, and date-window queries may include or exclude it based on updated_at. Use created_at as the earliest-availability timestamp, and treat headline, summary and symbols as possibly revised later (REST appears to return only the latest version).
- News limit is at most 50 per page and defaults to only 10. With symbols empty it returns all market news. Pagination tokens are opaque (they encode updated_at nanoseconds|id).
- Paper trading does NOT simulate dividends. Paper P&L on income-paying names will diverge from live and from total-return backtests.
- Unauthenticated calls return 401 (text/html on the data host) even for nonexistent paths. Don't read a 401 as 'endpoint exists', and handle non-JSON error bodies in the requests-based client. The FAQ also describes 403 for bad credentials or missing entitlement, with a JSON body like {code, message}.
- Rate limits: 200 per minute for historical data on the free plan and a separate 200 per minute per account on the Trading API. Throttle using X-RateLimit-Remaining and X-RateLimit-Reset (epoch) and back off on 429. Pulling a large universe with many paginated calls hits this quickly.

### UNKNOWN (not verifiable, so the code must fail safe)

- How date-only start/end (YYYY-MM-DD) are interpreted: UTC midnight or New York midnight, and whether end=YYYY-MM-DD includes that day's daily bar (t=04:00Z/05:00Z). Use full RFC-3339 timestamps with explicit offsets until tested.
- Maximum number of symbols per /v2/stocks/bars request. The deprecated v1 API documented 200; the v2 docs state no limit. URL length may be the practical cap.
- Whether symbols with no bars are omitted from the multi-symbol 'bars' map or returned as empty arrays.
- Exact first available dates: SIP 'since 2016' (exact first day not stated) and IEX 'from 2020' (from a staff forum post, not docs).
- Which feed is actually used for historical bars on the Basic plan when feed is omitted (schema says sip, FAQ says best available for the subscription).
- Whether a Paper Only (non-brokerage, non-US) account can technically query feed=sip history, and whether its terms allow it (the docs say IEX only).
- Whether news start/end filter on created_at or updated_at (only sort and pagination are shown to use updated_at).
- Whether news headline, summary, content or symbols are revised after creation, and whether any revision history is available. Only the presence of updated_at hints at edits.
- How far back /v1/corporate-actions history goes, and whether it has any max date range per request (the deprecated v2 announcements endpoint had 90 days).
- Which date field /v1/corporate-actions start/end filter on. The docs only say results are 'sorted by their process_date'; ex_date filtering is not documented.
- Whether the legacy /v1beta1/corporate-actions endpoint still works (it returns 401 without a key, but so does any path).
- Whether the corporate-actions SSE stream (/v1beta1/events/corporate-actions) is available to Basic/Trading API users, and how far back since/since_id replay goes.
- How complete the coverage of delisted, acquired or bankrupt tickers is. There is no official docs statement; only staff forum posts from 2022 and 2025.
- How asof handles a ticker later reused by an unrelated company, and whether asof follows mergers or CUSIP changes (the docs discuss only name changes).
- Whether the rate-limit window is rolling or fixed per minute (X-RateLimit-Reset suggests fixed), and whether news and corporate-actions calls count against the same 200/min bucket as bars on Basic. The only source is a 2022 blog post saying news shares the plan's limit.
- Time zone of the legacy /v2/calendar open/close/session fields (presumably America/New_York; not stated).
- Whether 'boats' and 'overnight' feed history is available on the Basic plan.

## Alpaca Trading API (paper)

_Alpaca Trading API, PAPER environment: base URLs, auth, orders, positions, account, activities, clock/calendar, assets, paper-fill simulation, rate limits (checked 2026-09-24)_

1. **The paper and live Trading API base URLs are separate. Market data uses one shared host for both.** [verified]  
   Paper Trading API: https://paper-api.alpaca.markets. Live Trading API: https://api.alpaca.markets. Market Data for both paper and live: https://data.alpaca.markets. Every Trading API reference spec lists servers [{Paper: https://paper-api.alpaca.markets}, {Live: https://api.alpaca.markets}]. SDKs usually read APCA_API_BASE_URL=https://paper-api.alpaca.markets.  
   Source: https://docs.alpaca.markets/us/docs/authentication.md

2. **Paper and live keys are completely separate. Each paper account also has its own keys.** [verified]  
   Quote: 'you cannot use your live account's credentials with the paper API, or vice versa.' The paper account gets a different API key from the live account, and the API spec is otherwise identical. Paper accounts are now created and deleted from the dashboard instead of reset: 'Don't forget to generate new API keys for any newly created account.' The default starting balance is $100k. The balance cannot be changed after creation.  
   Source: https://docs.alpaca.markets/us/docs/paper-trading.md

3. **Authentication uses a key pair in headers or HTTP Basic. OAuth client-credentials is not available for the Trading API.** [verified]  
   Headers: 'APCA-API-KEY-ID: <key id>' and 'APCA-API-SECRET-KEY: <secret>'. Alternative: HTTP Basic with the key ID as username and the secret as password. Callout: 'The Client Credentials authentication flow is not yet available for Trading API.' Live unauthenticated probe on 2026-09-24: every paper endpoint returned HTTP 401 with WWW-Authenticate: Bearer, Basic realm="alpaca.markets". /v2/account returned application/json {"message": "unauthorized."}. /v2/clock, /v2/calendar and /v3/clock returned a text/html nginx '401 Authorization Required' page. The endpoints are reachable and need a key.  
   Source: https://docs.alpaca.markets/us/docs/authentication.md

4. **POST /v2/orders request schema (CreateOrderRequest)** [verified]  
   Required: type (market|limit|stop|stop_limit|trailing_stop) and time_in_force (day|gtc|opg|cls|ioc|fok). Other fields: symbol (symbol, asset ID or pair), qty (string), notional (string), side (buy|sell), limit_price (required for limit/stop_limit), stop_price (required for stop/stop_limit), trail_price or trail_percent (one is required for trailing_stop), extended_hours (bool, default false), client_order_id (string), order_class (simple|bracket|oco|oto|mleg|""), take_profit{limit_price}, stop_loss{stop_price, limit_price}, position_intent (buy_to_open|buy_to_close|sell_to_open|sell_to_close), legs (mleg only, <=4), advanced_instructions. Price increments: >= $1.00 allows 2 decimals, < $1.00 allows 4 decimals. Orders beyond these increments are rejected.  
   Source: https://docs.alpaca.markets/us/reference/postorder.md

5. **qty and notional are mutually exclusive. Fractional and notional orders are DAY-only, so they cannot use opg, cls, ioc, fok or gtc.** [verified]  
   Spec: notional is a 'dollar amount to trade. Cannot work with qty. Can only work for market order types and day for time in force.' The fractional-trading page says market, limit, stop and stop_limit are supported with time_in_force=day, and that sending both qty and notional returns a 400 error. Both fields accept up to 9 decimals. The fractional TIF table shows DAY = Yes for all four types, and GTC, IOC, FOK, OPG and CLS = No for all of them. Errors: 422 {"code":42210000,"message":"fractional orders must be DAY orders"} and 422 {"code":40010001,"message":"notional must be >= 1.00"}. Fractional short sells are rejected with 422 'fractional orders cannot be sold short'. The asset must have fractionable=true, otherwise the order is rejected ('requested asset is not fractionable').  
   Source: https://docs.alpaca.markets/us/docs/orders-at-alpaca.md

6. **client_order_id: at most 128 characters, auto-generated when omitted, and a duplicate returns 422.** [verified]  
   Spec: 'A unique identifier for the order. Automatically generated if not sent. (<= 128 characters)', maxLength 128. Duplicate response: HTTP 422 {"code":40010001,"message":"client_order_id must be unique"}. The Alpaca Learn errors article describes this as a duplicate 'used for another active order'. Look an order up with GET /v2/orders:by_client_order_id?client_order_id=<id> (required query param). The path uses a colon; it is not /v2/orders/{id}.  
   Source: https://docs.alpaca.markets/us/reference/getorderbyclientorderid.md

7. **On an order-submit timeout, Alpaca says not to resend.** [verified]  
   FAQ: after a timeout, 'The order may have been sent to the market for execution. You should not attempt to resend the order or mark the timed-out order as canceled until confirmed.' In practice, use a deterministic client_order_id and reconcile through GET /v2/orders:by_client_order_id before retrying.  
   Source: https://docs.alpaca.markets/us/docs/working-with-orders.md

8. **Time-in-force semantics and cutoff windows (ET)** [verified]  
   day: valid in regular hours 9:30-16:00 ET and canceled if unfilled after the closing auction. If submitted after the close it is queued for the next trading day. gtc: auto-canceled 90 days after creation. The cancel job runs at 4:15pm ET on the date in the order's expires_at field, and the order can stay in pending_cancel until the venue confirms. opg: MOO/LOO in the opening auction only. Rejected if submitted after 9:28am and before 7:00pm ET; after 7:00pm it is queued for the next day's open. cls: MOC/LOC. Rejected if submitted after 3:50pm and before 7:00pm ET; after 7:00pm it is queued for the next day's close. ioc: any unfilled part is canceled. fok: all or nothing. Orders not eligible for extended hours that are submitted after 4:00pm ET are queued for the next trading day. The whole-share TIF table marks IOC, FOK, OPG and CLS as 'Yes*', and the footnote says 'Please contact the sales team for any TIF marked with a*'.  
   Source: https://docs.alpaca.markets/us/docs/orders-at-alpaca.md

9. **Extended-hours orders must be limit orders with DAY or GTC. Bracket, OCO and OTO orders cannot use extended hours.** [verified]  
   Sessions: overnight 8:00pm-4:00am ET (Sun-Fri), pre-market 4:00am-9:30am, after-hours 4:00pm-8:00pm. Set extended_hours=true, type=limit and time_in_force day or gtc. Anything else returns 422 'extended hours order must be DAY or GTC limit orders'. Bracket orders: 'extended_hours must be false or omitted.' Trailing stops accept only day or gtc and do not trigger outside regular hours. Asset attributes 'overnight_tradable' and 'fractional_eh_enabled' control overnight and fractional extended-hours eligibility.  
   Source: https://docs.alpaca.markets/us/docs/orders-at-alpaca.md

10. **Bracket, OTO and OCO order rules** [verified]  
   bracket: order_class='bracket', take_profit.limit_price, stop_loss.stop_price and optional stop_loss.limit_price. TIF must be day or gtc. For a buy, the take-profit limit must be above the stop-loss stop. Cancelling one leg cancels the rest. If the take-profit partially fills, the stop-loss is reduced to the remaining quantity. oto: entry plus exactly one of take_profit or stop_loss. Replacing an OTO is not yet supported. oco: exit orders only (the entry is already filled), type='limit'. The stop price must be at least $0.01 away from the base price. Legs are reported as separate orders unless nested=true, which groups them under 'legs'.  
   Source: https://docs.alpaca.markets/us/docs/orders-at-alpaca.md

11. **The OrderStatus enum has 17 values. The POST response usually shows accepted or pending_new, not a fill.** [verified]  
   Enum: new, partially_filled, filled, done_for_day, canceled, expired, replaced, pending_cancel, pending_replace, accepted, pending_new, accepted_for_bidding, stopped, rejected, suspended, calculated, held. 'held' is in the enum but is not described in the lifecycle table. The docs call filled, canceled, expired and replaced final ('no further updates'), and describe rejected the same way. 'accepted' means received but not yet routed and is 'often seen outside of trading session hours'. An order can be canceled until it reaches filled, canceled or expired. The spec's POST response examples show status 'accepted' or 'pending_new'.  
   Source: https://docs.alpaca.markets/us/docs/orders-at-alpaca.md

12. **Order object timestamp fields are UTC with nanosecond precision** [verified]  
   Fields: id, client_order_id, created_at, updated_at, submitted_at, filled_at, expired_at, expires_at, canceled_at, failed_at, replaced_at, replaced_by, replaces, asset_id, symbol, asset_class, notional, qty, filled_qty, filled_avg_price, order_class, type (order_type is deprecated in favour of type), side, position_intent, time_in_force, limit_price, stop_price, trail_price, trail_percent, hwm, extended_hours, legs, status. Example: "filled_at": "2022-04-19T17:45:05.024916716Z". Numbers are JSON strings. A local test showed Python 3.14.6 datetime.fromisoformat parses these strings but truncates to microseconds.  
   Source: https://docs.alpaca.markets/us/reference/postorder.md

13. **Error codes and body format for order submission** [verified]  
   POST /v2/orders: 403 'Buying power or shares is not sufficient', 422 'Input parameters are not recognized'. The body is JSON {"code": <int>, "message": "..."}, for example 40010001 and 42210000. Other documented messages: 403 'insufficient buying power', 403 'account is not authorized to trade', 403 'account is restricted to liquidation only', 403 {"code":40310100,"message":"trade denied due to pattern day trading protection"} (legacy), 422 'invalid time_in_force'.  
   Source: https://alpaca.markets/learn/how-to-fix-common-trading-api-errors-at-alpaca

14. **Wash-trade protection rejects some opposite-side orders with 403, including in paper** [verified]  
   Every new order is checked against open orders in the same symbol. Examples: an existing market buy with a new sell of any type is always rejected. An existing limit buy with a new limit sell is rejected when the buy limit >= the sell limit. Bracket, OCO and trailing-stop orders are exempt. Quote: 'Our wash trade protection also applies to your paper trading account.'  
   Source: https://docs.alpaca.markets/us/docs/user-protection.md

15. **GET /v2/orders defaults to open orders and 50 results** [verified]  
   Params: status open|closed|all (default open). limit: default 50, max 500. after and until filter on submission time and are exclusive. direction asc|desc: default desc, ordered by submission time. nested (bool). symbols (comma-separated). side. asset_class (array). before_order_id / after_order_id: mutually exclusive with each other and must not be combined with after/until. DELETE /v2/orders cancels all open orders and returns 207 Multi-Status, with a per-order 500 for orders that can no longer be canceled.  
   Source: https://docs.alpaca.markets/us/reference/getallorders-1.md

16. **PATCH /v2/orders/{order_id} (replace) creates a new order with a new ID and can fail after returning success** [verified]  
   The response is 'The new Order object with the new order ID', and the body can include a new client_order_id. 'A success return code from a replaced order does NOT guarantee the existing open order has been replaced': if the old order fills first, the new order is rejected and the events arrive on trade_updates. Orders cannot be replaced while accepted, pending_new, pending_cancel or pending_replace. Non-IPO notional orders cannot be replaced at all, and fractional qty cannot change. During a replace, buying power is reduced by the larger of the old and new orders.  
   Source: https://docs.alpaca.markets/us/reference/patchorderbyorderid-1.md

17. **Positions endpoints and fields** [verified]  
   GET /v2/positions and GET /v2/positions/{symbol_or_asset_id}. Required fields: asset_id, symbol, exchange, asset_class, avg_entry_price, qty, side (long|short), market_value, cost_basis, unrealized_pl, unrealized_plpc, unrealized_intraday_pl, unrealized_intraday_plpc, current_price, lastday_price, change_today, asset_marginable. Also qty_available (qty minus shares tied up in open orders). The *_plpc and change_today fields are fractions ('by a factor of 1'), not percentages. 'Once a position is closed, it will no longer be queryable through this API.'  
   Source: https://docs.alpaca.markets/us/reference/getallopenpositions.md

18. **GET /v2/account fields and definitions** [verified]  
   Required fields: id and status. Other fields: cash; equity (= cash + long_market_value + short_market_value); buying_power (multiplier 4: (last_equity - last_maintenance_margin)*4; multiplier 2: max(equity - initial_margin, 0)*2; multiplier 1: cash); regt_buying_power; non_marginable_buying_power; options_buying_power; multiplier (string '1'|'2'|'4'); initial_margin; maintenance_margin; last_maintenance_margin; last_equity ('Equity as of previous trading day at 16:00:00 ET'); balance_asof (date of the last_* snapshot); long_market_value; short_market_value; sma; shorting_enabled; trading_blocked; account_blocked; transfers_blocked; trade_suspended_by_user; accrued_fees; pending_reg_taf_fees; created_at; crypto_status. portfolio_value is deprecated and equal to equity. status enum: INACTIVE, PAPER_ONLY, ONBOARDING, SUBMISSION_FAILED, SUBMITTED, ACCOUNT_UPDATED, APPROVAL_PENDING, ACTIVE, REJECTED, ACCOUNT_CLOSED, APPROVED, ACCOUNT_CLOSED_PENDING, ACTION_REQUIRED, LIMITED. The current schema does NOT list pattern_day_trader, daytrade_count or daytrading_buying_power, and none of the downloaded pages mention them.  
   Source: https://docs.alpaca.markets/us/reference/getaccount-1.md

19. **Alpaca docs say the Pattern Day Trader rule has been replaced by FINRA's Intraday Margin Rule** [verified]  
   Quote: 'The definition of a "pattern day trader" has been eliminated.' There is no $25k minimum and day trades are unlimited. The Reg T $2,000 minimum still applies to margin. The user-protection page (updated 2026-09-16) lists 'Intraday Margin Rule' where PDT protection used to be. An intraday margin deficit must be met within 2 business days, with a 90-day freeze if unmet by the fifth. Per the FINRA explainer page, firms may use either regime during a 12-month transition after SEC approval.  
   Source: https://docs.alpaca.markets/us/docs/the-intraday-margin-rule.md

20. **Account activities: endpoints, FILL fields, pagination** [verified]  
   GET /v2/account/activities and GET /v2/account/activities/{activity_type}, for example /FILL. Params: activity_types (comma list; cannot be combined with category), category (trade_activity|non_trade_activity), order_id (uuid), date, after, until (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ; these filter on created_at, 'not the activity's settlement date'), direction (asc|desc, default desc), page_size, page_token (= id of the last item on the previous page). page_size: if date is not given, default and max are 100; if date is given, all results are returned with no max. FILL (TradingActivity) fields: activity_type='FILL', id ('<timestamp>::<uuid>', e.g. '20190524113406977::8efc7b9a-...'), cum_qty, leaves_qty, price, qty, side, symbol, transaction_time (UTC, e.g. '2019-05-24T15:34:06.977Z'), order_id, order_status, type ('fill'|'partial_fill'). Non-trade activities (e.g. DIV) have date, net_amount, per_share_amount, qty, cusip, status (executed|correct|canceled), and their creation date is 'typically the day after the trade date (in UTC)'.  
   Source: https://docs.alpaca.markets/us/reference/getaccountactivities-2.md

21. **GET /v2/clock returns timestamps with an ET offset, not UTC. A newer /v3/clock also exists.** [verified]  
   /v2/clock fields: timestamp, is_open, next_open, next_close. Example: "timestamp": "2025-06-24T14:15:22-04:00", "next_close": "2025-06-24T16:00:00-04:00". GET /v3/clock takes markets (comma-separated codes such as NYSE, NASDAQ, IEX, BOATS) and time (date-time, to evaluate the clock at a given instant). It returns {clocks:[{market:{acronym,name,timezone e.g. 'America/New_York',mic,bic}, timestamp, is_market_day, next_market_open, next_market_close, phase (closed|pre|core|lunch|post), phase_until}]}.  
   Source: https://docs.alpaca.markets/us/reference/legacyclock.md

22. **GET /v2/calendar covers 1970-2029 and returns times as bare strings with no timezone** [verified]  
   Params: start and end (inclusive), date_type TRADING (default) or SETTLEMENT. Fields: date 'YYYY-MM-DD', open 'HH:MM' (e.g. '09:30'), close 'HH:MM' ('16:00'), session_open 'HHMM' ('0400'), session_close 'HHMM' ('2000'), settlement_date. Early closures are reflected. The newer GET /v3/calendar/{market} takes start (default today), end (default start + 1 week) and timezone=UTC (optional; otherwise the market's timezone). It returns date, core_start, core_end, pre_start, pre_end, post_start, post_end, lunch_start, lunch_end and settlement_date as RFC3339 values with offsets (e.g. '2025-01-02T09:30:00-05:00').  
   Source: https://docs.alpaca.markets/us/reference/legacycalendar.md

23. **GET /v2/assets returns inactive assets unless you filter. borrow_status is the documented borrow field.** [verified]  
   Params: status (e.g. 'active'; 'By default, all statuses are included'), asset_class (default us_equity), exchange (AMEX|ARCA|BATS|NYSE|NASDAQ|NYSEARCA|OTC|CRYPTO), attributes (comma list; matches assets with any of them). Single asset: GET /v2/assets/{symbol_or_asset_id}. Required fields: id, class, exchange, symbol, name, status (active|inactive), tradable, marginable, shortable, fractionable. Optional: borrow_status (easy_to_borrow|hard_to_borrow), cusip (nullable), attributes, margin_requirement_long and margin_requirement_short (decimal strings). maintenance_margin_requirement is deprecated. The boolean 'easy_to_borrow' appears in examples but is not in the schema properties. Some assets are data-only and have tradable=false.  
   Source: https://docs.alpaca.markets/us/reference/get-v2-assets-1.md

24. **Asset attributes enum** [verified]  
   ptp_no_exception, ptp_with_exception, ipo (limit orders only before the first secondary-market trade), has_options (still set if the asset only had expired contracts in the past), options_late_close, fractional_eh_enabled, overnight_tradable, overnight_halted.  
   Source: https://docs.alpaca.markets/us/reference/get-v2-assets-1.md

25. **How paper fills are simulated** [verified]  
   Paper orders are matched against the current NBBO. 'Your order quantity is not checked against the NBBO quantities', so fills can exceed real liquidity. When an order is eligible to fill, it gets a random-size partial fill 10% of the time, and the remainder is re-evaluated. Limit orders fill only when marketable: buy limit >= best ask, sell limit <= best bid. Not simulated: market impact, information leakage, latency slippage, queue position, price improvement, regulatory fees, dividends. Borrow fees are marked 'Coming Soon'. Paper accounts send no fill emails.  
   Source: https://docs.alpaca.markets/us/docs/paper-trading.md

26. **Paper-only accounts get IEX market data only** [verified]  
   Quote: 'As an Alpaca Paper Only Account holder, you are only entitled to receive and make use of IEX market data.' Paper fills are still matched against the NBBO whatever the account type.  
   Source: https://docs.alpaca.markets/us/docs/paper-trading.md

27. **How buying power is checked at order time** [verified]  
   Open buy-long and sell-short orders reduce available buying power until they fill or are canceled. Sell-long and buy-to-cover orders do not restore buying power until they execute. The price used for the check: far side of the NBBO during the core session, inside midpoint during extended hours, latest cached trade when both are closed. An opening short is valued at MAX(limit, 3% above ask) × qty.  
   Source: https://docs.alpaca.markets/us/docs/orders-at-alpaca.md

28. **Trading API rate limit is 200 requests per minute per account, returning 429 when exceeded** [verified]  
   Support article (Dec 2022): 'the API is throttled, currently 200 requests per minute, per account'. Going over returns '429 - Too Many Requests', and users cannot raise the limit themselves. The current OpenAPI specs list 429 responses and the headers X-RateLimit-Limit ('Request limit per minute'), X-RateLimit-Remaining and X-RateLimit-Reset ('The UNIX epoch when the remaining quota changes').  
   Source: https://alpaca.markets/support/usage-limit-api-calls

29. **The trade_updates websocket is the recommended source of order state. The paper stream sends binary frames.** [verified]  
   URL: wss://paper-api.alpaca.markets/stream (live: wss://api.alpaca.markets/stream). Send {"action":"auth",...} with the key pair, then {"action":"listen","data":{"streams":["trade_updates"]}}. 'The trade_updates stream coming from wss://paper-api.alpaca.markets/stream uses binary frames.' JSON and MessagePack are supported. fill and partial_fill events carry timestamp, price, qty, position_qty (signed) and execution_id. Other events: new, canceled, expired, replaced, done_for_day, accepted, pending_new, rejected, pending_cancel, pending_replace, calculated, order_replace_rejected, order_cancel_rejected.  
   Source: https://docs.alpaca.markets/us/docs/websocket-streaming.md

### Gotchas

- Paper does NOT simulate dividends. A paper equity curve understates total return against any total-return benchmark or backtest, and DIV activities will be missing. Adjust for dividends separately or compare on a price-return basis.
- Paper fills ignore displayed size: any quantity fills at the NBBO with no impact. Capacity and slippage in paper are therefore optimistic. Model costs explicitly and do not calibrate slippage from paper fills.
- Paper-only accounts see only IEX data, but paper orders fill against the NBBO. Signals built from IEX quotes can differ from the reference price used for fills.
- Point-in-time: day orders submitted after 4:00pm ET are queued for the next trading day. opg orders are rejected from 9:28am to 7:00pm ET, and cls orders from 3:50pm to 7:00pm ET. A signal computed after the close cannot fill at that close. The backtest must assume execution at the next open or later.
- Fractional and notional orders are DAY-only, so there is no MOO/MOC (opg/cls), gtc or ioc for them. Notional orders cannot be replaced; cancel and resubmit.
- A POST 200 response usually carries status 'accepted' or 'pending_new'. Treat it only as acknowledgement. Track fills through trade_updates or by polling. Partial fills happen randomly on 10% of fill events in paper, and a partially filled DAY order can end as canceled with filled_qty > 0.
- Idempotency: on a timeout, do NOT blindly resend. Use a deterministic client_order_id (max 128 characters) and check GET /v2/orders:by_client_order_id. A 422 'client_order_id must be unique' means an order with that ID already exists (it may have been accepted). It does not mean the order never reached the broker.
- GET /v2/orders defaults to status=open and limit=50 (max 500). Without status=all or closed and pagination (until/after, or before_order_id/after_order_id), closed or older orders are silently left out.
- The time filters point at different fields: /v2/orders after/until use submission time (exclusive), while /v2/account/activities after/until/date use created_at, not settlement date. Non-trade activities such as fees or dividends are typically created the day after the trade date (UTC).
- Account activities page_size has a default and max of 100 when date is not given. Paginate with page_token = the id of the last item. If a date is given, all results come back in one response.
- Timezones differ by endpoint: /v2/clock returns ET-offset timestamps (-04:00/-05:00), order and activity timestamps are UTC 'Z' with nanoseconds, and /v2/calendar open/close are bare 'HH:MM' strings (session_open/close are 'HHMM') with no zone. Normalise explicitly. Python fromisoformat truncates nanoseconds to microseconds.
- /v2/calendar only covers 1970-2029. Queries beyond 2029 will return nothing.
- /v2/assets returns inactive assets by default; pass status=active and check tradable=true. Asset flags (tradable, shortable, fractionable, borrow_status, marginable) are current snapshots with no as-of parameter. Building a historical universe from them introduces survivorship and look-ahead bias.
- The boolean 'easy_to_borrow' appears only in examples. The documented field is borrow_status (easy_to_borrow|hard_to_borrow). Code that reads asset['easy_to_borrow'] may fail or get None.
- The current account schema no longer documents pattern_day_trader, daytrade_count or daytrading_buying_power, and the docs say PDT has been replaced by the Intraday Margin Rule. Code that reads these fields should use .get() and must not depend on them.
- Wash-trade protection (403) applies in paper: an opposite-side order that could interact with an existing open order in the same symbol is rejected. For example, a resting limit sell plus a new market buy is rejected. Rebalancers must cancel conflicting orders first or use bracket/OCO.
- Broker-held protective stops (execution/protective_stops.py) are standalone `type=stop`, `time_in_force=gtc` SELL orders for whole shares, priced at 2 decimals (4 below $1), per items 4, 5 and 7 above. UNVERIFIED and NOT relied on: that shares reserved by an open sell order are unavailable to a second sell ('insufficient qty available'). The code never tests this. Every exit first cancels the resting stop and waits for a CONFIRMED cancel, and it sends nothing if the cancel is not confirmed. Also UNVERIFIED: how Alpaca treats open GTC orders across a split. The runner re-derives the stop from the ledger's split record and replaces it, and it never places a stop at or above the current price.
- Open buy orders reserve buying power until they fill or are canceled, and sells do not free buying power until they execute. Buy orders in a rebalance sent right after sells can get 403 insufficient buying power.
- A PATCH replace returns a NEW order id and can still fail after returning 200 if the original fills first. Always reconcile via the stream or by polling. Orders cannot be replaced while accepted, pending_new, pending_cancel or pending_replace.
- GTC orders are auto-canceled 90 days after creation at 4:15pm ET on expires_at, and may stay in pending_cancel for a while.
- 401 error bodies can be an HTML nginx page (clock and calendar) rather than JSON. Do not call response.json() blindly on errors.
- Deleting and recreating a paper account (the replacement for reset) issues new keys. Stale keys will return 401.

### UNKNOWN (not verifiable, so the code must fail safe)

- Whether client_order_id uniqueness covers ALL historical orders or only active/open ones. The Learn article says 'another active order', but the reference spec only says 'unique'.
- Whether the PAPER environment currently enforces the Intraday Margin Rule, legacy PDT protection, or neither, and whether /v2/account in paper still returns pattern_day_trader or daytrade_count in practice. Verifying this needs an authenticated call, which was not made.
- Whether paper simulates stock splits and other corporate actions. The docs only say dividends are not simulated.
- How paper fills opg/cls orders: at the real auction print or at the NBBO around 9:30/16:00. This is not documented.
- Whether paper-only accounts' fills use the SIP NBBO or IEX quotes. The docs say 'NBBO' for all accounts but also say paper-only accounts get IEX data only.
- Whether the 200 requests/min rate limit (from a 2022 support article) is still current, whether it is per key or per account, and whether paper and live share a bucket. The current reference docs list rate-limit headers but no number (the header example shows 100).
- Timezone of the /v2/calendar open/close/session_open/session_close fields. Presumably ET, but the spec does not say.
- What the 'Yes*' / 'contact the sales team' footnote on whole-share IOC, FOK, OPG and CLS means for self-directed Trading API paper users.
- How much order and account-activity history is retained, and whether deleted paper accounts' history can still be retrieved.
- Whether paper enforces hard-to-borrow locates or short availability. Borrow fees are marked 'Coming Soon' for paper.
- How inactive or delisted assets are kept in /v2/assets over time. There is no documented as-of or history parameter.

## SEC EDGAR

_SEC EDGAR public APIs (data.sec.gov submissions / companyfacts / companyconcept / frames, bulk ZIPs, ticker files) for point-in-time fundamentals and earnings-event timing, plus Fama-French SIC industry mapping_

1. **data.sec.gov APIs need no authentication or API key, and they return JSON.** [verified]  
   Base host: https://data.sec.gov. Covered XBRL forms: 10-Q, 10-K, 8-K, 20-F, 40-F, 6-K and their variants. The docs say CORS is not supported. However, a live response on 2026-09-24 included the header Access-Control-Allow-Origin: *. This does not matter for server-side Python.  
   Source: https://www.sec.gov/search-filings/edgar-application-programming-interfaces

2. **Fair-access rate limit: no more than 10 requests per second per user, counted across all machines. If you exceed it, your IP is throttled until your rate stays below the threshold for 10 minutes.** [verified]  
   Quote: 'no more than 10 requests per second, regardless of the number of machines used to submit requests. If a user or application submits more than 10 requests per second, further requests from the IP address(es) may be limited for a brief period. Once the rate of requests has dropped below the threshold for 10 minutes, the user may resume.' Unclassified bots are not allowed.  
   Source: https://www.sec.gov/about/privacy-information#security

3. **Required User-Agent format: a company or app name plus a contact email. The SEC also recommends sending Accept-Encoding: gzip, deflate.** [verified]  
   SEC sample headers: 'User-Agent: Sample Company Name AdminContact@<sample company domain>.com', 'Accept-Encoding: gzip, deflate', 'Host: www.sec.gov'. Live test: a request with curl's default UA got HTTP 403 and an HTML page titled 'Your Request Originates from an Undeclared Automated Tool'. A generic 'Mozilla/5.0' UA sent to www.sec.gov/files/company_tickers_exchange.json got HTTP 403 with an HTML page titled 'Request Rate Threshold Exceeded', even though only one request was made. A declared 'Name email' UA returned 200.  
   Source: https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data

4. **Submissions endpoint: https://data.sec.gov/submissions/CIK##########.json, where the CIK is zero-padded to 10 digits. An unpadded CIK returns 404.** [verified]  
   Live: CIK0000320193.json returned 200. CIK320193.json returned 404, with an S3-style XML body (<Error><Code>NoSuchKey</Code>), not JSON. Every 404 on data.sec.gov (unknown concept, unknown frame, bad CIK) is application/xml NoSuchKey.  
   Source: https://data.sec.gov/submissions/CIK0000320193.json

5. **Top-level fields in the submissions JSON.** [verified]  
   Keys (live, AAPL): cik (10-digit string '0000320193'), entityType ('operating'), sic (string '3571'), sicDescription ('Electronic Computers'), ownerOrg, insiderTransactionForOwnerExists, insiderTransactionForIssuerExists, name, tickers (list, e.g. ['AAPL']), exchanges (parallel list, e.g. ['Nasdaq']), ein, lei, description, website, investorWebsite, category ('Large accelerated filer'), fiscalYearEnd ('MMDD' string, '0926'), stateOfIncorporation, stateOfIncorporationDescription, addresses{mailing,business}, phone, flags, formerNames[{name, from, to} as ISO timestamps], filings{recent, files}.  
   Source: https://data.sec.gov/submissions/CIK0000320193.json

6. **filings.recent is a set of parallel (columnar) arrays. It holds at least one year of filings or the 1,000 most recent filings, whichever is more.** [verified]  
   Column names (live): accessionNumber, filingDate (YYYY-MM-DD), reportDate (YYYY-MM-DD or ''), acceptanceDateTime, act, form, fileNumber, filmNumber, items, core_type, size, isXBRL, isInlineXBRL, isXBRLNumeric, primaryDocument, primaryDocDescription. For AAPL, recent had exactly 1000 rows covering 2015-07-27 to 2026-09-22.  
   Source: https://www.sec.gov/search-filings/edgar-application-programming-interfaces

7. **filings.files lists older pages of the history. Each page is a flat columnar object with the same keys as recent, but without the 'recent' wrapper.** [verified]  
   AAPL files = [{name:'CIK0000320193-submissions-001.json', filingCount:1249, filingFrom:'1994-01-26', filingTo:'2015-07-25'}]. Fetch it at https://data.sec.gov/submissions/CIK0000320193-submissions-001.json. It returned 1249 rows, but the actual maximum filingDate on the page was 2015-07-24. Treat filingTo as approximate. To get the full history, concatenate recent with every page.  
   Source: https://data.sec.gov/submissions/CIK0000320193-submissions-001.json

8. **acceptanceDateTime in the submissions JSON is true UTC. The 'Z' suffix is accurate.** [verified]  
   Format: 'YYYY-MM-DDTHH:MM:SS.000Z'. Checked against the SGML header <ACCEPTANCE-DATETIME>, which the SEC FAQ says is Eastern time. 10-Q 0000320193-24-000081: header 20240801180334 (EDT) = JSON 2024-08-01T22:03:34.000Z. 8-K 0000320193-26-000005: header 20260129163033 (EST) = JSON 2026-01-29T21:30:33.000Z. The offset moves with daylight saving time, so convert with zoneinfo('America/New_York'). Never apply a fixed -5h.  
   Source: https://www.sec.gov/Archives/edgar/data/320193/000032019324000081/0000320193-24-000081-index-headers.html

9. **Filing-date rule (17 CFR 232.13(a)(2)): a submission that begins after 5:30 p.m. ET is deemed filed on the next business day. Forms 3/4/5, Schedule 14N, Form 144 and Schedules 13D/13G have a 10 p.m. ET cutoff, as do Rule 462(b) registration statements.** [verified]  
   Live examples. AAPL 10-Q accn 0000320193-24-000081: accepted 2024-08-01 18:03:34 ET, filingDate 2024-08-02, header 'FILED AS OF DATE: 20240802' but 'DATE AS OF CHANGE: 20240801'. 8-K 0001140361-25-025275: accepted 2025-07-08 21:18 ET (JSON 2025-07-09T01:18:13Z), filingDate 2025-07-09. In both cases acceptanceDateTime still shows the real acceptance instant; only filingDate rolls forward.  
   Source: https://www.ecfr.gov/current/title-17/chapter-II/part-232/section-232.13

10. **EDGAR accepts filings Monday to Friday, 6:00 a.m. to 10:00 p.m. ET. Filings are usually on sec.gov within 1-3 minutes of acceptance, but the SEC does not guarantee this lag.** [verified]  
   Some submissions that begin after 5:30 p.m. ET (10 p.m. for Forms 3/4/5) are disseminated the next business day. The SEC FAQ says: 'There is no timestamp to indicate when filing content is first available on sec.gov.' The submissions API typically updates within 1 second of dissemination and the XBRL APIs within 1 minute. Both can lag at peak times.  
   Source: https://www.sec.gov/about/webmaster-frequently-asked-questions

11. **The 'items' column is a comma-separated string of 8-K item numbers. Earnings releases carry item 2.02.** [verified]  
   AAPL earnings 8-Ks show items '2.02,9.01'. Filing headers confirm 'ITEM INFORMATION: Results of Operations and Financial Condition' for 2.02. Filter with form in {'8-K','8-K/A'} and '2.02' in items.split(','). For 8-Ks, reportDate is the event date. Example of timing: AAPL 8-K 2.02 accepted 16:30 ET (20:30Z in EDT, 21:30Z in EST) on the same day as filingDate, which is after the 16:00 close.  
   Source: https://data.sec.gov/submissions/CIK0000320193.json

12. **The 8-K 2.02 earnings press release does NOT feed us-gaap numbers into companyfacts. The quarter's fundamentals arrive in the API only when the 10-Q/10-K is filed, usually later.** [verified]  
   AAPL: the 8-K 2.02 was accepted 2026-07-30 20:30Z and the 10-Q 0000320193-26-000020 was accepted 2026-07-31 10:01Z (06:01 ET). The only 8-K-sourced us-gaap facts in AAPL companyfacts come from two recast-financials 8-Ks (0001193125-13-170623 and 0001193125-15-023732). Earnings 8-Ks are XBRL-tagged (isXBRL=1) for the cover page only.  
   Source: https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json

13. **companyfacts structure: {cik:int, entityName, facts:{taxonomy:{Concept:{label, description, units:{unit:[fact,...]}}}}}.** [verified]  
   Each fact has keys start (duration concepts only), end, val, accn, fy, fp, form, filed, and optional frame. The frame key is omitted when it doesn't apply; it is never null. For AAPL NetIncomeLoss, 252 facts had no frame and 86 had one. Taxonomies seen: 'us-gaap', 'dei' (ifrs-full and srt are also possible per the docs). AAPL has 503 us-gaap concepts and 2 dei concepts (EntityCommonStockSharesOutstanding, EntityPublicFloat). Units are keys like 'USD', 'shares', 'USD/shares'.  
   Source: https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json

14. **'filed' is a date only ('YYYY-MM-DD') with no time. It equals the submissions filingDate for the same accn, including rolled-forward after-hours dates.** [verified]  
   Joined all 15,013 AAPL facts whose accn appears in submissions.recent: 100% filed == filingDate. Example: the 10-Q accepted 2024-08-01 18:03 ET has filed='2024-08-02'. For intraday availability, join accn to submissions.acceptanceDateTime. Note that 10,033 AAPL facts had accns only in the older submissions page, so the join needs all pages.  
   Source: https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json

15. **fy/fp describe the FILING that reported the fact (its fiscal year and period), not the fact's own period. They can be null.** [verified]  
   Example: the FY2007 net income fact (start 2006-10-01, end 2007-09-29) carries fy=2009, fp='FY' because it came from the FY2009 10-K. Facts from 8-Ks have fy=None and fp=None (569 in AAPL). Identify periods by (start, end) only. fp values: FY, Q1, Q2, Q3, None. There is no Q4; a fiscal Q4 must be derived as FY minus the 9-month YTD.  
   Source: https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json

16. **The same (concept, start, end) appears once per filing that reported it, as original, comparative, or restated/reclassified values with later filed dates. The value can change.** [verified]  
   Examples. AAPL AccountsPayableCurrent @2017-09-30: 49,049M in the 10-K filed 2017-11-03 and three 10-Qs, then 44,242M in the 10-K filed 2018-11-05. AccruedLiabilitiesCurrent @2008-09-27: 3,719M (10-K 2009-10-27), then 4,224M (10-K/A 2010-01-25). 340 of 12,366 AAPL (concept, unit, period) groups have differing values. Within one accn there were 0 duplicate (start, end) facts. Point-in-time rule: keep facts with filed (better, acceptance) <= as-of; choose earliest-filed for 'as first reported' or latest-filed-so-far for 'best known at as-of'.  
   Source: https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json

17. **'frame' is attached to the LAST-FILED fact for each calendar period. It therefore carries LOOK-AHEAD bias and must not be used as a point-in-time selector.** [verified]  
   The SEC says the frames API 'aggregates one fact for each reporting entity that is last filed that most closely fits the calendrical period'. Live check on AAPL: in 9,716 of 9,716 framed period groups, the frame sits on the max-filed fact. Example: CY2007 net income frame is on the 10-K/A value 3,495M filed 2010-01-25, not the 3,496M filed 2009-10-27. In the frames API, Abbott's CY2025Q2 revenue points to accn 0001628280-26-050134, a 2026 filing.  
   Source: https://www.sec.gov/search-filings/edgar-application-programming-interfaces

18. **Frames period syntax: CY#### is annual (365±30 days), CY####Q# is quarterly (91±30 days), CY####Q#I is an instant. The URL unit uses '-per-' for ratio units.** [verified]  
   URL: https://data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json, e.g. /us-gaap/EarningsPerShareDiluted/USD-per-shares/CY2024.json → 200. Using 'USD/shares' in the path → 404, even though companyconcept/companyfacts label the unit 'USD/shares'. The default XBRL unit is 'pure'. Periods are matched by best fit, so non-calendar fiscal years land in a CY frame: AAR's FY 2024-06-01..2025-05-31 appears in CY2024.  
   Source: https://www.sec.gov/search-filings/edgar-application-programming-interfaces

19. **Frames response shape: {taxonomy, tag, ccp, uom, label, description, pts, data:[{accn, cik:int, entityName, loc, start?, end, val}]}. There is NO 'filed' field and only one fact per entity.** [verified]  
   Live: CY2025Q2 RevenueFromContractWithCustomerExcludingAssessedTax had pts=2564. Assets CY2025Q4I had pts=6152, and its instant rows have no 'start'. loc looks like 'US-IL'. To date a frames row you must join accn to submissions. There are no pagination tokens; one JSON holds all entities.  
   Source: https://data.sec.gov/api/xbrl/frames/us-gaap/RevenueFromContractWithCustomerExcludingAssessedTax/USD/CY2025Q2.json

20. **Standalone fiscal-Q4 frames are sparse because 10-K filers generally report only the annual duration.** [verified]  
   Live pts counts: NetIncomeLoss CY2024 = 6058, but CY2024Q4 = 1255. 10-Qs report both 3-month and YTD (6M/9M) durations for the same end date; e.g. the AAPL Q3 10-Q has start 2023-10-01 (9M) and start 2024-03-31 (3M), both ending 2024-06-29. Tell them apart by (end - start) days.  
   Source: https://data.sec.gov/api/xbrl/frames/us-gaap/NetIncomeLoss/USD/CY2024Q4.json

21. **companyconcept endpoint: https://data.sec.gov/api/xbrl/companyconcept/CIK##########/{taxonomy}/{tag}.json.** [verified]  
   Response keys (live): cik (int), taxonomy, tag, label, description, entityName, units{unit:[facts with the same fields as companyfacts]}. AAPL EarningsPerShareDiluted has unit key 'USD/shares'. A concept the company never used returns 404 XML NoSuchKey, not an empty JSON.  
   Source: https://data.sec.gov/api/xbrl/companyconcept/CIK0000320193/us-gaap/EarningsPerShareDiluted.json

22. **The XBRL APIs include only non-custom-taxonomy facts that apply to the entire filing entity. Company-extension tags and dimensional (segment/class) facts are excluded.** [verified]  
   Doc: the APIs aggregate facts that 'Use a non-custom taxonomy (e.g. us-gaap, ifrs-full, dei, or srt)' and 'Apply to the entire filing entity'. Consequence (live): Alphabet (CIK 1652044, multiple share classes) has NO dei:EntityCommonStockSharesOutstanding in companyfacts, only EntityPublicFloat. dei EntityCommonStockSharesOutstanding CY2024Q4I frame pts = 2831, versus 6268 for Assets.  
   Source: https://www.sec.gov/search-filings/edgar-application-programming-interfaces

23. **Revenue concept names change over time and between companies, so a fallback chain that picks the concept with the most recent 'end' is required.** [verified]  
   AAPL: SalesRevenueNet was filed 2009-07-22..2018-08-01; Revenues appears only in the 2018-11-05 10-K; RevenueFromContractWithCustomerExcludingAssessedTax from 2019-01-30. Alphabet: RevenueFromContractWithCustomerExcludingAssessedTax last end 2025-03-31, then Revenues through 2026-06-30. Frames CY2024: Revenues pts 2503, RevenueFromContract...ExcludingAssessedTax pts 2967, SalesRevenueNet 404 (no longer used). SalesRevenueNet CY2016 pts 1915.  
   Source: https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json

24. **Concept names for other core fundamentals were confirmed to exist in live frames/companyfacts, with CY2024 coverage counts.** [verified]  
   NetIncomeLoss (6058; ProfitLoss including NCI 3316), EarningsPerShareDiluted unit USD/shares (5536), Assets instant (6268), StockholdersEquity (6011; StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest 2327), LongTermDebt (2216) and LongTermDebtNoncurrent (1820), OperatingIncomeLoss (5016; absent for many filers), GrossProfit (2626; e.g. absent for Alphabet), WeightedAverageNumberOfDilutedSharesOutstanding, CommonStockSharesOutstanding (balance-sheet date), dei:EntityCommonStockSharesOutstanding (cover-page date).  
   Source: https://data.sec.gov/api/xbrl/frames/us-gaap/NetIncomeLoss/USD/CY2024.json

25. **dei:EntityCommonStockSharesOutstanding is a cover-page instant. Its 'end' is the cover date (weeks after the period end), not the balance-sheet date.** [verified]  
   AAPL: end 2026-07-17 from the 10-Q for the period ending 2026-06-27 (filed 2026-07-31), frame CY2026Q2I. Unit key 'shares'.  
   Source: https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json

26. **XBRL history starts in 2009. Earlier periods appear only as comparatives in 2009+ filings, so first-reported values before 2009 are not available.** [verified]  
   AAPL NetIncomeLoss: earliest filed 2009-07-22, earliest period end 2007-09-29. The SEC doc says XBRL 'was first required by the SEC in 2009'.  
   Source: https://www.sec.gov/search-filings/edgar-application-programming-interfaces

27. **Bulk nightly ZIP archives exist for companyfacts and submissions (about 1.4 GB and 1.6 GB). The docs say they are republished around 3:00 a.m. ET.** [verified]  
   https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip: HEAD returned Content-Length 1,409,389,023 bytes, Last-Modified Thu 24 Sep 2026 04:24:14 GMT. https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip: 1,565,294,470 bytes, Last-Modified 04:31:44 GMT. The observed timestamps are about 00:30 ET, earlier than the documented ~3 a.m. ET. companyfacts.zip holds all the data for the Frames and Company Facts APIs.  
   Source: https://www.sec.gov/search-filings/edgar-application-programming-interfaces

28. **company_tickers_exchange.json has the shape {fields:['cik','name','ticker','exchange'], data:[[cik:int, name, ticker, exchange],...]}.** [verified]  
   Live: 10,461 rows. Exchange counts: Nasdaq 4361, NYSE 3301, OTC 2538, CBOE 44, null 217. 1447 CIKs have more than one ticker (e.g. 1067983 has BRK-B and BRK-A; share classes use a hyphen). Last-Modified was 2026-09-22 21:37:29 GMT, with Cache-Control max-age=42. The companion https://www.sec.gov/files/company_tickers.json is a dict keyed '0','1',... of {cik_str:int, ticker, title}. The SEC says it updates both 'periodically' and does 'not guarantee accuracy or scope'.  
   Source: https://www.sec.gov/files/company_tickers_exchange.json

29. **EDGAR header timestamp semantics: ACCEPTANCE-DATETIME is in Eastern time. FILED AS OF DATE is the official filing date and can be reset by a Post Acceptance Correction. DATE AS OF CHANGE is the date of the last PAC.** [verified]  
   The SEC FAQ defines Acceptance Time as 'Time (EST) at which the submission was accepted by EDGAR. Format is HHMMSS'. Header files are at /Archives/edgar/data/{cik}/{accn-no-dashes}/{accn}-index-headers.html. Full and quarterly indexes are rebuilt weekly on Saturday to include PAC deletes and updates.  
   Source: https://www.sec.gov/about/webmaster-frequently-asked-questions

30. **Submissions sic and fiscalYearEnd are current-only snapshots. Each filing's SGML header keeps the values as they were at filing time.** [verified]  
   The 10-Q header 0000320193-24-000081 shows 'FISCAL YEAR END: 0928' and 'STANDARD INDUSTRIAL CLASSIFICATION: ELECTRONIC COMPUTERS [3571]'. The current submissions JSON shows fiscalYearEnd '0926' (52/53-week year). tickers and exchanges are also current only; formerNames carries name history with from/to dates.  
   Source: https://www.sec.gov/Archives/edgar/data/320193/000032019324000081/0000320193-24-000081-index-headers.html

31. **Accession number format is 10-digit filer/agent CIK, then a 2-digit year, then a sequence number. The prefix is often a filing agent, not the issuer.** [verified]  
   Example 0001140361-26-015711 is an AAPL 8-K filed by an agent. Never infer the issuer CIK from the accn prefix.  
   Source: https://www.sec.gov/about/webmaster-frequently-asked-questions

32. **Kenneth French's SIC-to-industry definition files are ZIPs at a fixed URL pattern.** [verified]  
   https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes12.zip (881 bytes) and .../Siccodes49.zip (9,587 bytes), both Last-Modified 2020-01-08. Siccodes48.zip also exists. Linked as 'Download industry definitions' from Data_Library/det_12_ind_port.html and det_49_ind_port.html. Per a third-party parser, the same pattern also serves 5, 10, 17, 30 and 38.  
   Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_49_ind_port.html

33. **Fama-French industry construction uses the SIC code that applied at the time: Compustat SIC for the fiscal year ending in t-1, else CRSP SIC, with assignment at the end of June of year t.** [verified]  
   The page states: 'We assign each NYSE, AMEX, and NASDAQ stock to an industry portfolio at the end of June of year t based on its four-digit SIC code at that time.'  
   Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_12_ind_port.html

34. **Siccodes text format (from a third-party parser, not the file itself): an industry header line, then indented SIC range lines.** [UNVERIFIED]  
   Parser regexes: header r'(\d+)\s+([A-z]+)' (industry number, short name, then a long description); range r'(\d+)-(\d+) ?(.*)' (4-digit start-end inclusive plus an optional sub-description). Industries with no ranges (the residual 'Other') are skipped by that parser, so the residual must be assigned as the fallback.  
   Source: https://raw.githubusercontent.com/stoffprof/ff_industries/main/ff_industries/ff_industries.py

### Gotchas

- LOOK-AHEAD via frame: the 'frame' key (and the whole frames API) sits on the LAST-filed fact for a period, and it moves when a restatement or later comparative is filed. Point-in-time code must ignore frame and choose facts by filed/acceptance <= as_of.
- Frames API rows carry no filed date, only accn. Using frames for historical cross-sections gives restated/latest values, not as-reported ones.
- Same-day look-ahead: 'filed'/filingDate is a date with no time. AAPL's earnings 8-K is accepted at 16:30 ET with filingDate the same day, which is after the close. Treating filed-date facts as known at that day's close or open leaks information. Use submissions acceptanceDateTime (UTC) converted to America/New_York and roll to the next session if it is after 16:00 ET.
- After-5:30pm-ET filings get filingDate = next business day, while acceptanceDateTime shows the true earlier instant. The UTC calendar date can be a day later than the ET date (e.g. 2025-07-09T01:18Z = 2025-07-08 21:18 ET). Always convert with a DST-aware zone, never date(UTC).
- acceptanceDateTime in the submissions JSON is genuine UTC ('Z'), but the SGML header ACCEPTANCE-DATETIME is Eastern local time with no offset. Mixing the two sources without conversion shifts times by 4-5 hours.
- fy/fp belong to the reporting filing, not the fact's period. Keying on fy/fp mislabels comparatives (e.g. FY2007 data tagged fy=2009). 8-K-sourced facts have fy/fp = null. Key on (start, end) instead.
- 10-Qs contain both 3-month and YTD durations with the same 'end'. Q4 is rarely reported standalone. Distinguish by duration days and derive Q4 = FY - 9M YTD. Beware 52/53-week years (91±30-day tolerance).
- Concepts switch over time (SalesRevenueNet → Revenues → RevenueFromContractWithCustomerExcludingAssessedTax, and back to Revenues for Alphabet in 2025). Picking the first concept found silently freezes the series at an old date. Choose per period across a fallback chain.
- dei:EntityCommonStockSharesOutstanding is missing for multi-class issuers (dimensional facts are excluded; e.g. Alphabet), and its 'end' is the cover date, not the balance-sheet date.
- 404s from data.sec.gov are XML (NoSuchKey), and the 403 for an undeclared UA or rate limiting is an HTML page. A requests client that calls .json() without checking status and content-type will crash or, worse, get swallowed. Treat 404 as 'concept not used', not as an error to retry.
- CIKs must be zero-padded to 10 digits in the URLs; an unpadded CIK gives 404. company_tickers files give cik as int (cik_str is also an int despite its name).
- Survivorship bias: company_tickers(_exchange).json, and tickers/exchanges/sic/fiscalYearEnd in the submissions JSON, are CURRENT snapshots only. Delisted names are absent, and historical SIC and fiscal-year-end must come from the per-filing SGML headers.
- filings.recent is capped (1 year or 1000 filings). Heavy Form 4/144 filers push old 10-K/10-Q/8-Ks into filings.files pages, and companyfacts accn joins silently miss unless all pages are loaded.
- Exceeding 10 req/s throttles the IP until the rate stays below the limit for 10 minutes. The limit is per user across all machines, so parallel workers must share one global rate limiter.
- Earnings 8-K (Item 2.02) numbers are not in companyfacts. XBRL fundamentals for a quarter appear only when the 10-Q/10-K is filed, often a day or more after the earnings event. Features built on companyfacts must be timestamped to the 10-Q/10-K acceptance, not to the earnings date.
- The accession-number prefix is often a filing agent's CIK, not the issuer's.
- Post Acceptance Corrections can change FILED AS OF DATE or remove filings after the fact, so historical indexes may not reflect them.

### UNKNOWN (not verifiable, so the code must fail safe)

- Exactly when a filing accepted after 5:30 p.m. ET first became publicly visible (same evening or next business day): the SEC says some such filings are disseminated the next business day, and there is no timestamp for first public availability on sec.gov.
- The exact tie-break when several facts for the same period share the latest filed date (e.g. a 10-Q and a 10-K/A filed the same day), meaning which one gets the frame.
- Whether companyfacts ever drops superseded or corrected facts after a Post Acceptance Correction or deletion. Not tested.
- The actual line format inside Siccodes12.txt/Siccodes49.txt was not read from the primary ZIP. It was not downloaded, per the file-download policy; the format comes from a third-party parser. Also not verified from the primary file: that FF12 industry 12 'Other' is a residual with no explicit ranges.
- The Content-Length figures for the bulk ZIPs are a single snapshot (2026-09-24). Size grows over time, and the uncompressed size was not checked.
- Whether data.sec.gov enforces a separate or stricter limit than 10 req/s, or returns Retry-After headers when throttling. Not observed.
- The meaning and allowed values of the submissions columns 'core_type', 'isXBRLNumeric' and 'act'. They appear in live data but are not documented on the API page.
- Whether the generic-UA 403 ('Request Rate Threshold Exceeded') was caused by the UA string or by shared-IP history. It was observed once, and the cause is not documented.

## SEC EDGAR Form 4 + House Clerk PTRs (congress / insider context, docs/ALT-DATA.md)

Verified live 2026-10-02 (`data/providers/sec_form4.py`, `data/providers/house_ptr.py`):
* `https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&...&output=atom` returns an Atom
  feed; each Form 4 appears once per filer role (issuer / reporting), so de-duplicate on the accession
  in `<id>` (`accession-number=...`). `<updated>` carries an explicit offset (e.g. `-04:00`).
* `https://www.sec.gov/Archives/edgar/data/<cik>/<accession-no-dashes>/<accession>.txt` (complete
  submission) holds `<ACCEPTANCE-DATETIME>YYYYMMDDHHMMSS` in the SGML header and the
  `<ownershipDocument>` XML. The header time is **US Eastern** local time: read as Eastern it equals the
  Atom `<updated>` instant exactly (checked on accession 0001161697-26-000229: 21:40:11 EDT =
  01:40:11Z).
* Daily form index `https://www.sec.gov/Archives/edgar/daily-index/<Y>/QTR<q>/form.<YYYYMMDD>.idx`:
  fixed-width text, published in the evening; 404 on weekends/holidays and before publication.
* Same User-Agent requirement and fair-access limit as the section above; one shared limiter.
* House Clerk `https://disclosures-clerk.house.gov/public_disc/financial-pdfs/2026FD.zip`: no key, no
  documented limit; `2026FD.xml` inside, `<Member>` rows with Prefix/Last/First/Suffix/FilingType/
  StateDst/Year/FilingDate (`M/D/YYYY`)/DocID. On 2026-10-02: 1,724 filings, 404 PTRs (FilingType `P`),
  357 with an electronic-style DocID (`2xxxxxxx`), latest PTR filing date 2026-09-30.
* PTR PDFs `https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/<YEAR>/<DocID>.pdf`; paper filings
  (DocID `8xxxxxx`/`9xxxxxx`) are scanned images without a text layer. The PTR text parser has only been
  exercised on a fixture shaped like the published layout, not yet on a live PDF.
* Senate eFD (`efdsearch.senate.gov`) refuses automated access (403): not used, not bypassed.

## LLM providers: structured output

_2026 LLM provider APIs (Anthropic, OpenAI, Google Gemini) for strict JSON-schema output via plain HTTP (requests): endpoints, auth, request/response shape, schema enforcement, usage/caching fields, errors/retry, current model IDs_

1. **Anthropic Messages API: endpoint, auth and required headers** [verified]  
   POST https://api.anthropic.com/v1/messages. Headers: 'x-api-key: <key>', 'anthropic-version: 2023-06-01' (the only current version; required on raw HTTP), 'content-type: application/json'; optional 'anthropic-beta: <ids>' for beta features. Live unauthenticated POST returned HTTP 401 {"type":"error","error":{"type":"authentication_error","message":"x-api-key header is required"},"request_id":"req_..."} plus headers 'request-id' and 'x-should-retry: false'.  
   Source: https://platform.claude.com/docs/en/api/versioning

2. **Anthropic request body fields for system prompt, messages, output cap, sampling** [verified]  
   Required: model, max_tokens (int, min 0; 0 = cache pre-warm only), messages (array of {role:'user'|'assistant', content: string | content blocks}). system = string OR array of {type:'text', text, cache_control?}. Optional: stop_sequences, stream, metadata, tools, tool_choice, thinking, output_config {effort, format}, service_tier ('auto'|'standard_only'), inference_geo, cache_control (top-level auto-caching). temperature is DEPRECATED: for models released after Claude Opus 4.6 only the value 1.0 is accepted, any other value -> 400; top_p only >=0.99 accepted; top_k any value -> 400.  
   Source: https://platform.claude.com/docs/en/api/messages/create

3. **Anthropic native structured output: output_config.format (GA, no beta header)** [verified]  
   Body: "output_config": {"format": {"type": "json_schema", "schema": {<JSON Schema, every object needs additionalProperties:false>}}}. Result is valid JSON in the response's text content block (select content[] item with type=='text'). Old top-level 'output_format' param and beta header 'structured-outputs-2025-11-13' still accepted for a transition period but deprecated. Supported models include claude-fable-5-1, claude-opus-5-5, claude-opus-5, claude-sonnet-5, claude-haiku-4-5-20251001 (and opus-4-8/4-7/4-6, sonnet-4-6, sonnet-4-5, opus-4-5). Incompatible with citations (400) and assistant prefill.  
   Source: https://platform.claude.com/docs/en/build-with-claude/structured-outputs

4. **Anthropic strict tool use as alternative schema enforcement** [verified]  
   Tool definition: {"name","description","input_schema":{...additionalProperties:false, required:[...]},"strict": true} (strict is top-level on the tool, not on tool_choice). Constrained decoding guarantees tool_use.input validates. tool_choice options: {type:'auto', disable_parallel_tool_use?}, {type:'any'}, {type:'tool', name}, {type:'none'}.  
   Source: https://platform.claude.com/docs/en/api/messages/create

5. **Forced tool_choice (the classic 'force a tool to get JSON' trick) returns 400 on Claude Opus 5.5 and Claude Fable 5.1** [verified]  
   tool_choice {type:'any'} or {type:'tool',name:...} -> 400 'tool_choice: type "tool" and "any" are not supported for this model.' (also on count_tokens). Use output_config.format, or tool_choice auto + strict:true + prompt instruction. Opus 5.5 also rejects assistant prefill (400) and thinking {type:'disabled'} / budget_tokens (400); thinking is always on, default effort 'medium'.  
   Source: https://platform.claude.com/docs/en/models/opus-5-5/migration-guide

6. **Anthropic structured-output JSON Schema subset and hard limits** [verified]  
   Supported: object/array/string/integer/number/boolean/null, enum (primitives only), const, anyOf, allOf (not with $ref), internal $ref/$defs/definitions, default, string formats date-time,time,date,duration,email,hostname,uri,ipv4,ipv6,uuid, array minItems 0 or 1 only, simple regex pattern. NOT supported (400): recursive schemas, minimum/maximum/multipleOf, minLength/maxLength, other array constraints, additionalProperties != false. Per-request limits across all strict schemas: 20 strict tools, 24 optional (non-required) params total, 16 union-type params (anyOf or type arrays). Else 400 'Schema is too complex for compilation'; compile timeout 180 s. Compiled grammars cached 24 h from last use (first use adds latency).  
   Source: https://platform.claude.com/docs/en/build-with-claude/structured-outputs

7. **Anthropic structured outputs can still be non-conformant in specific cases** [verified]  
   stop_reason 'refusal' (HTTP 200, billed, output may not match schema; stop_details {type:'refusal', category: 'cyber'|'bio'|'frontier_llm'|'reasoning_extraction'|'general_harms'|null, explanation}); stop_reason 'max_tokens' (truncated JSON). Enum/const casing NOT guaranteed (may differ only in capitalization, no error). Output property order: required properties first (schema order), then optional ones.  
   Source: https://platform.claude.com/docs/en/build-with-claude/structured-outputs

8. **Anthropic response shape and stop_reason values** [verified]  
   Response Message: id, type:'message', role:'assistant', model, content[] (block types incl. text, thinking, redacted_thinking, tool_use, server_tool_use...), stop_reason in {end_turn, max_tokens, stop_sequence, tool_use, pause_turn, refusal, model_context_window_exceeded}, stop_sequence, stop_details (non-null only for refusal), usage. On always-thinking models (Opus 5.5, Fable 5.1) content[0] can be a 'thinking' block (empty text by default); grammar applies only to the final text, not thinking.  
   Source: https://platform.claude.com/docs/en/api/messages/create

9. **Anthropic usage fields and cached-token semantics** [verified]  
   usage: input_tokens (ONLY tokens after the last cache breakpoint, i.e. excludes cache reads/writes), cache_creation_input_tokens, cache_read_input_tokens, cache_creation {ephemeral_5m_input_tokens, ephemeral_1h_input_tokens}, output_tokens (includes thinking), output_tokens_details {thinking_tokens <= output_tokens}, server_tool_use {web_search_requests, web_fetch_requests}, service_tier ('standard'|'priority'|'batch'), inference_geo. Total input = input_tokens + cache_creation_input_tokens + cache_read_input_tokens. In streaming, input/cache counts arrive in message_start; message_delta usage is cumulative.  
   Source: https://platform.claude.com/docs/en/api/messages/create

10. **Anthropic prompt caching mechanics** [verified]  
   Explicit: cache_control {type:'ephemeral'} (optional 'ttl':'1h') on a block; up to 4 breakpoints; or top-level cache_control for automatic placement. Default TTL 5 min (refreshed free on hit); write cost 1.25x base (5m) or 2x (1h); reads 0.1x base (0.05x on Opus 5.5, 0.025x on Fable 5.1). Minimum cacheable prefix: 512 tokens (Fable 5.1, Opus 5.5, Opus 5), 1,024 (Sonnet 5, Opus 4.8), 4,096 (Haiku 4.5); below minimum it silently does not cache (both cache fields 0, no error). 20-block lookback window. Changing output_config.format invalidates the prompt cache.  
   Source: https://platform.claude.com/docs/en/build-with-claude/prompt-caching

11. **Anthropic error codes and retry signals** [verified]  
   Error body {type:'error', error:{type,message}, request_id}. 400 invalid_request_error (also returned when an org/workspace spend limit you set is reached), 401 authentication_error, 402 billing_error, 403 permission_error, 404 not_found_error (bad or unavailable model ID), 409 conflict_error, 413 request_too_large (32 MB Messages limit, returned by Cloudflare), 429 rate_limit_error, 500 api_error, 504 timeout_error, 529 overloaded_error. SSE streams can emit an error event after HTTP 200.  
   Source: https://platform.claude.com/docs/en/api/errors

12. **Anthropic rate-limit headers and 429 behaviour** [verified]  
   429 includes 'retry-after' (seconds). Headers: anthropic-ratelimit-{requests,tokens,input-tokens,output-tokens}-{limit,remaining,reset}; *-reset values are RFC 3339 timestamps; token remaining rounded to nearest thousand. Token-bucket replenishment. ITPM counts input_tokens + cache_creation_input_tokens but NOT cache_read_input_tokens (most models). OTPM ignores max_tokens. Monthly tier spend-cap 429 has NO retry-after and fails until 00:00 UTC on the 1st of next month. Sharp traffic ramps can trigger 'acceleration limit' 429s.  
   Source: https://platform.claude.com/docs/en/api/rate-limits

13. **Anthropic current model IDs (live docs, 2026-09-24)** [verified]  
   Docs say start with claude-opus-5-5 (Claude Opus 5.5, released 2026-09-22, $4/$20 per MTok, 1M ctx, 128K out, default effort medium, reliable knowledge cutoff Jun 2026). Top tier: claude-fable-5-1 ($10/$50, cutoff Jun 2026). Mid: claude-sonnet-5 ($2/$10, cutoff Jan 2026, default effort high). Cheap: claude-haiku-4-5-20251001 (alias claude-haiku-4-5; $1/$5; 200K ctx, 64K out; extended thinking via budget_tokens, no effort param; reliable cutoff Feb 2025, training cutoff Jul 2025). Docs state dateless IDs from the 4.6 generation on are pinned snapshots. All four IDs in the brief are confirmed valid.  
   Source: https://platform.claude.com/docs/en/about-claude/models/overview

14. **Claude Haiku 4.5 has a near-term retirement floor** [verified]  
   claude-haiku-4-5-20251001: status Active, retirement 'Not sooner than October 15, 2026' (about 3 weeks after 2026-09-24). Sonnet 5 not sooner than 2027-06-30; Opus 5.5 not sooner than 2027-09-22; Fable 5.1 not sooner than 2027-09-01.  
   Source: https://platform.claude.com/docs/en/about-claude/model-deprecations

15. **OpenAI Responses API endpoint, auth and core request fields** [verified]  
   POST https://api.openai.com/v1/responses, header 'Authorization: Bearer <key>' (live unauthenticated call -> 401 'Missing bearer or basic authentication in header', response header x-request-id). Fields: model, input (string or message array; roles incl. system/developer/user), instructions (system/developer message; NOT carried over when using previous_response_id), max_output_tokens (caps visible + reasoning tokens), temperature (0-2), top_p, reasoning {effort}, text {format, verbosity}, store, truncation ('disabled' default -> 400 on overflow; 'auto' silently drops earliest items), prompt_cache_key, prompt_cache_retention, prompt_cache_options, service_tier, safety_identifier.  
   Source: https://developers.openai.com/api/reference/resources/responses/methods/create

16. **OpenAI strict JSON schema on Responses API** [verified]  
   "text": {"format": {"type": "json_schema", "name": "<a-zA-Z0-9_- max 64>", "schema": {...}, "strict": true, "description"?: "..."}}. {type:'json_object'} is legacy JSON mode (valid JSON, no schema adherence). Chat Completions equivalent: POST https://api.openai.com/v1/chat/completions with "response_format": {"type":"json_schema","json_schema":{"name","schema","strict":true,"description"?}}; output cap there is max_completion_tokens (max_tokens deprecated, not compatible with o-series).  
   Source: https://developers.openai.com/api/docs/guides/structured-outputs

17. **OpenAI strict-mode schema rules and limits** [verified]  
   All fields must be listed in 'required' (emulate optional with type ["T","null"]); additionalProperties:false on every object; root must be an object and not anyOf. Supported: string pattern/format (date-time,time,date,duration,email,hostname,ipv4,ipv6,uuid), number multipleOf/maximum/exclusiveMaximum/minimum/exclusiveMinimum, array minItems/maxItems, anyOf, $defs, recursive schemas. Unsupported: allOf, not, dependentRequired, dependentSchemas, if/then/else. Limits: 5,000 object properties, 10 nesting levels, 120,000 chars of names/enum/const strings, 1,000 enum values (15,000 chars for a single string enum with >250 values). Output key order follows schema order. Unsupported schema with strict:true -> error.  
   Source: https://developers.openai.com/api/docs/guides/structured-outputs

18. **OpenAI Responses output parsing over raw HTTP** [verified]  
   Raw response has 'output' array of items (reasoning items, message items, tool calls...). Find item type=='message' then content[] part type=='output_text' (.text) or type=='refusal' (.refusal). 'output_text' at top level is an SDK-only convenience property (Python/JS SDKs) and is NOT reliable for requests-based clients. status in {completed, failed, in_progress, cancelled, queued, incomplete}; incomplete_details.reason in {max_output_tokens, max_messages, content_filter, steered}. Chat Completions: choices[].message.content / .refusal, finish_reason in {stop, length, tool_calls, content_filter, function_call}.  
   Source: https://developers.openai.com/api/reference/resources/responses/methods/create

19. **OpenAI usage fields (cached tokens INCLUDED in input count)** [verified]  
   Responses usage: input_tokens, input_tokens_details {cached_tokens, cache_write_tokens}, output_tokens, output_tokens_details {reasoning_tokens}, total_tokens. Ordinary input = input_tokens - cached_tokens - cache_write_tokens. Chat Completions usage: prompt_tokens, completion_tokens, total_tokens, prompt_tokens_details {cached_tokens, cache_write_tokens, audio_tokens,...}, completion_tokens_details {reasoning_tokens, accepted_prediction_tokens, rejected_prediction_tokens, audio_tokens}.  
   Source: https://developers.openai.com/api/docs/guides/prompt-caching

20. **OpenAI prompt caching is automatic; GPT-5.6+ now bills cache writes** [verified]  
   Enabled by default. GPT-5.6 and later: minimum 1,024 visible input tokens; implicit (default) or explicit mode via prompt_cache_options.mode ('implicit'|'explicit') + prompt_cache_breakpoint {mode:'explicit'} on content blocks (not allowed in top-level instructions); up to 4 cache writes per request; prompt_cache_options.ttl only '30m'. Cache writes cost 1.25x uncached input; reads 0.1x. Earlier models: prompt_cache_retention 'in_memory' (~5-10 min idle, up to 1 h) or '24h'; cached_tokens rounded down to multiple of 128; prompt_cache_key matters for routing before GPT-5.6.  
   Source: https://developers.openai.com/api/docs/guides/prompt-caching

21. **OpenAI error codes and retry semantics** [verified]  
   429 variants distinguished by error.code: rate limit reached (retry), 'slow_down' (rate_limit_error; ramp too fast), 'credit_balance_exhausted', 'organization_spend_limit_exceeded', 'project_spend_limit_exceeded', 'organization_usage_limit_exceeded' (error.type may be insufficient_quota) - retrying billing/quota 429s never succeeds. 503 service_unavailable_error / code 'server_is_overloaded' (Python SDK raises InternalServerError, not RateLimitError). 500 server error. 401 invalid key/IP not allowlisted; 403 unsupported country. 'Retry-After' (seconds, treat as minimum) may be present on temporary 429/503; else exponential backoff with jitter.  
   Source: https://developers.openai.com/api/docs/guides/error-codes

22. **OpenAI rate-limit headers use duration strings, not timestamps** [verified]  
   x-ratelimit-limit-requests, x-ratelimit-limit-tokens, x-ratelimit-remaining-requests, x-ratelimit-remaining-tokens, x-ratelimit-reset-requests (e.g. '1s'), x-ratelimit-reset-tokens (e.g. '6m0s'), plus x-ratelimit-{limit,remaining,reset}-project-tokens when project limits apply. Ramp guidance: above 1M TPM grow no more than 50% per 15 min.  
   Source: https://developers.openai.com/api/docs/guides/rate-limits

23. **OpenAI current recommended model IDs** [verified]  
   Flagship: gpt-6-astra ($10 in / $1 cached / $12.5 cache write / $50 out per 1M; 1,050,000 ctx, 922K max input, 128K out; knowledge cutoff 2026-04-30; reasoning.effort low|medium|high|xhigh|max, NO 'none' -> 400). Balanced: gpt-5.6-terra ($2/$12, cutoff 2026-02-16). Cost tier: gpt-5.6-luna ($0.20/$1.20, cutoff 2026-02-16). Also gpt-6-sol ($2/$10, cutoff 2026-04-20) and gpt-6-luna ($0.10/$0.50, cutoff 2026-05-18); both support effort 'none' (default medium). Prompts >272K input tokens priced 2x input and 1.5x output for the whole request. All support structured_outputs; Chat Completions and Responses both supported.  
   Source: https://developers.openai.com/api/docs/models

24. **OpenAI sampling params and tool calling constraints on current reasoning models** [verified]  
   When reasoning effort is not 'none', remove temperature, top_p, top_logprobs (Chat: also logprobs). GPT-6 Astra tool calling requires Responses API (Chat Completions unsupported for its function calling); GPT-6 Sol/Luna support function calling in Chat Completions only with reasoning_effort 'none'.  
   Source: https://developers.openai.com/api/docs/guides/latest-model

25. **OpenAI max_output_tokens can be exhausted by hidden reasoning** [verified]  
   If reasoning + output hits max_output_tokens or the context limit, status='incomplete', incomplete_details.reason='max_output_tokens', possibly with NO visible output while still billing input + reasoning tokens. OpenAI recommends reserving >=25,000 tokens for reasoning+output when starting.  
   Source: https://developers.openai.com/api/docs/guides/reasoning

26. **OpenAI stores responses by default** [verified]  
   Responses API 'store' defaults to true; stored response data retained for at least 30 days. Chat Completions stored by default for new accounts. Set store:false to disable (in stateless mode reasoning items carry encrypted_content).  
   Source: https://developers.openai.com/api/reference/resources/responses/methods/create

27. **Gemini generateContent endpoint and auth** [verified]  
   POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent with header 'x-goog-api-key: <key>' (docs also show ?key= query param - avoid, it puts the key in URLs/logs). Body: contents[] ({role:'user'|'model', parts:[{text}]}), systemInstruction (Content, text only), generationConfig, tools, toolConfig, safetySettings, cachedContent ('cachedContents/{id}'), serviceTier, store, labels. Unauthenticated live call to /v1beta/models -> 403 {"error":{"code":403,"message":...,"status":"PERMISSION_DENIED"}}. Google now recommends the new Interactions API (POST /v1beta/interactions, response_format array) for new work but states generateContent 'remains fully supported'.  
   Source: https://ai.google.dev/api/generate-content

28. **Gemini JSON-schema output on generateContent (current field is generationConfig.responseFormat)** [verified]  
   Current: "generationConfig": {"responseFormat": {"text": {"mimeType": "application/json", "schema": {<JSON Schema>}}}}. Legacy (marked deprecated in reference but still documented): generationConfig.responseMimeType='application/json' + responseSchema (OpenAPI-3.0 subset, uppercase types like 'OBJECT'), or _responseJsonSchema/responseJsonSchema (JSON Schema; mutually exclusive with responseSchema). Other generationConfig fields: maxOutputTokens (default = model output_token_limit), temperature, topP, topK, seed, stopSequences (max 5), candidateCount, thinkingConfig {thinkingLevel | thinkingBudget, includeThoughts} (thinkingLevel + thinkingBudget together -> 400).  
   Source: https://ai.google.dev/gemini-api/docs/generate-content/structured-output

29. **Gemini supported JSON Schema subset and non-strict behaviour** [verified]  
   Types string/number/integer/boolean/object/array/null (via type array); title, description; object properties/required/additionalProperties (bool or schema); string enum/format (date-time,date,time); number/integer enum/minimum/maximum; array items/prefixItems/minItems/maxItems; reference also lists $id,$defs,$ref,$anchor,anyOf,oneOf(=anyOf), propertyOrdering. Docs: 'The model ignores unsupported properties' and large/deeply nested schemas may be rejected. Output keys follow schema key order. No 'strict' flag exists; docs say JSON is syntactically valid but values must be validated.  
   Source: https://ai.google.dev/gemini-api/docs/generate-content/structured-output

30. **Gemini response shape and block/finish reasons** [verified]  
   GenerateContentResponse: candidates[] {content{parts[{text}], role:'model'}, finishReason, safetyRatings, index}, promptFeedback {blockReason, safetyRatings}, usageMetadata, modelVersion, responseId, modelStatus. If promptFeedback.blockReason is set (SAFETY, OTHER, BLOCKLIST, PROHIBITED_CONTENT, IMAGE_SAFETY) NO candidates are returned. finishReason enum: STOP, MAX_TOKENS, SAFETY, RECITATION, LANGUAGE, OTHER, BLOCKLIST, PROHIBITED_CONTENT, SPII, MALFORMED_FUNCTION_CALL, UNEXPECTED_TOOL_CALL, TOO_MANY_TOOL_CALLS, MISSING_THOUGHT_SIGNATURE, MALFORMED_RESPONSE, ESCALATION, PUP_LIMITED_DISABLED, image-specific values.  
   Source: https://ai.google.dev/api/generate-content

31. **Gemini usageMetadata fields (cached tokens INCLUDED in prompt count; thoughts separate from candidates)** [verified]  
   usageMetadata: promptTokenCount (includes cachedContent tokens), cachedContentTokenCount, candidatesTokenCount, thoughtsTokenCount, toolUsePromptTokenCount, totalTokenCount (= prompt + thoughts + candidates), promptTokensDetails[], cacheTokensDetails[], candidatesTokensDetails[], toolUsePromptTokensDetails[], serviceTier. Pricing page: output price 'including thinking tokens'.  
   Source: https://ai.google.dev/api/generate-content

32. **Gemini context caching** [verified]  
   Implicit caching on by default for Gemini 2.5+ (no cost-saving guarantee); minimum 4,096 tokens for Gemini 3.x (3.8/3.7/3.6/3.5 Flash, 3.1 Pro Preview), 2,048 for 2.5. Explicit caching (beta, v1beta only, generateContent only - not Interactions): POST /v1beta/cachedContents, reference via 'cachedContent'; default TTL 1 hour if unset; billed cached-input rate plus hourly storage (e.g. $0.50-$1.00 per 1M tokens per hour). Cache hits reported in usageMetadata.cachedContentTokenCount.  
   Source: https://ai.google.dev/gemini-api/docs/generate-content/caching

33. **Gemini errors, retries and quotas** [verified]  
   generateContent errors use google.rpc JSON {error:{code,message,status,details}}. Retryable: 429 RESOURCE_EXHAUSTED, 503 UNAVAILABLE, 408, other 5xx (500 INTERNAL, 504 DEADLINE_EXCEEDED); do not retry 400/402 (Prepay credits depleted)/403. Docs recommend exponential backoff with jitter (SDK: up to 4 retries, ~1 s initial, 60 s max). Rate limits (RPM, input TPM, RPD, sometimes TPD) are per PROJECT not per API key; RPD resets at midnight Pacific time.  
   Source: https://ai.google.dev/gemini-api/docs/troubleshooting

34. **Gemini current model IDs** [verified]  
   Flagship GA: gemini-3.8-flash (1,048,576 in / 65,536 out; default thinking level medium; levels low|medium|high - 'minimal' returns an error; introductory $0.75 in / $3.75 out per 1M through 2026-12-31, $1.50/$7.50 from 2027-01-01). Pro tier only in preview: gemini-3.1-pro-preview ($2/$12 <=200K prompt; knowledge cutoff Jan 2025 per Gemini 3 guide). Cheap: gemini-3.5-flash-lite ($0.30 in / $2.50 out) or gemini-3.1-flash-lite ($0.25/$1.50, cutoff Jan 2025). Gemini 2.5 models restricted to prior users; gemini-2.0-flash, 3-pro-preview, 3.1-flash-lite-preview shut down.  
   Source: https://ai.google.dev/gemini-api/docs/models

35. **Gemini 3.x sampling guidance** [verified]  
   Gemini 3 guide: keep temperature at default 1.0 (lower values may cause looping/degraded reasoning). Gemini 3.8 Flash migration checklist: strip temperature, top_p, top_k; replace thinking_budget with thinking_level; remove candidate_count (unsupported in Gemini 3+); remove prefilled model turns. Troubleshooting page still lists temperature range 0.0-1.0 while the reference says [0.0, 2.0] (inconsistent docs).  
   Source: https://ai.google.dev/gemini-api/docs/latest-model

36. **Prompt-caching support and cached-token reporting across the three providers** [verified]  
   All three support prompt caching. Anthropic: opt-in via cache_control, reported in usage.cache_read_input_tokens / cache_creation_input_tokens and EXCLUDED from usage.input_tokens. OpenAI: automatic, reported in usage.input_tokens_details.cached_tokens / cache_write_tokens (Chat: prompt_tokens_details.*) and INCLUDED in input_tokens/prompt_tokens. Gemini: implicit automatic + explicit cachedContents, reported in usageMetadata.cachedContentTokenCount and INCLUDED in promptTokenCount.  
   Source: https://platform.claude.com/docs/en/build-with-claude/prompt-caching

37. **Model knowledge cutoffs relevant to point-in-time backtests** [verified]  
   Claude Opus 5.5 and Fable 5.1: reliable and training cutoff Jun 2026; Sonnet 5: Jan 2026; Haiku 4.5: reliable Feb 2025 / training Jul 2025. GPT-6 Astra 2026-04-30; GPT-6 Sol 2026-04-20; GPT-6 Luna 2026-05-18; GPT-5.6 Terra/Luna 2026-02-16. Gemini 3.1 Pro Preview / 3.1 Flash-Lite Jan 2025 (Gemini 3 guide). Any LLM-derived signal computed for an as-of date before the model's training cutoff can embed knowledge of later outcomes.  
   Source: https://platform.claude.com/docs/en/about-claude/models/overview

38. **Anthropic long non-streaming requests** [verified]  
   Docs warn against large max_tokens without streaming or Batches (idle connections dropped); SDKs refuse non-streaming requests expected to exceed 10 minutes. For requests-based clients set explicit connect/read timeouts and TCP keep-alive, or use stream:true (SSE named events message_start, content_block_start/delta/stop, message_delta, message_stop; errors can arrive mid-stream after HTTP 200). Max output 128K (Opus 5.5/Fable 5.1/Sonnet 5), 64K Haiku 4.5.  
   Source: https://platform.claude.com/docs/en/api/errors

### Gotchas

- LOOK-AHEAD LEAK: every LLM has a training cutoff (Claude Opus 5.5/Fable 5.1 Jun 2026, GPT-6 Astra Apr 30 2026, GPT-5.6 Feb 16 2026, Gemini 3.1 Jan 2025). Using an LLM to score/extract from historical filings or news dated before its cutoff lets it 'know' what happened next. Store model ID + cutoff with every LLM-derived feature, treat only post-cutoff as-of dates as clean out-of-sample, and never enable server-side web search / Gemini Google Search grounding / URL context in backtest-time calls (they pull today's information into a historical as-of date).
- Non-reproducibility: temperature cannot be pinned on current flagships (Anthropic models after Opus 4.6 reject temperature != 1.0 with 400; OpenAI says remove temperature/top_p when reasoning effort != none; Gemini 3.8 Flash says strip temperature/top_p/top_k). Re-running a backtest will not give identical LLM outputs; persist every raw request/response keyed by (provider, model, prompt hash, as-of date) and replay from storage rather than re-querying.
- Record the model that actually served each call (Anthropic response 'model', OpenAI 'model', Gemini 'modelVersion'). Anthropic's server-side fallbacks (beta) and Opus 5.5 safety fallbacks can make a different model answer; Gemini/OpenAI aliases may move.
- Anthropic: the classic 'force JSON via tool_choice {type:tool}' pattern returns HTTP 400 on claude-opus-5-5 and claude-fable-5-1. Use output_config.format (json_schema) instead; tool_choice any/tool still works on claude-sonnet-5 and claude-haiku-4-5-20251001, so a client that works on Haiku silently breaks when switched to Opus 5.5.
- Anthropic: on always-thinking models content[0] is often a 'thinking' block (empty text by default); parsing content[0]['text'] raises KeyError or returns ''. Always select the block with type=='text'.
- Anthropic structured outputs: HTTP 200 with stop_reason 'refusal' or 'max_tokens' can return JSON that does NOT match the schema. Check stop_reason == 'end_turn' before json.loads; enum/const capitalization is not guaranteed (compare case-insensitively); numeric constraints (minimum/maximum) are rejected with 400, so range-check in code.
- Anthropic: a monthly tier spend-cap 429 has NO retry-after header and keeps failing until 00:00 UTC on the 1st of next month. A retry loop that waits on retry-after or backs off forever will hang; detect missing retry-after and fail fast. A spend limit you set yourself returns 400, not 429.
- OpenAI: several 429s are billing/quota errors (credit_balance_exhausted, *_spend_limit_exceeded, organization_usage_limit_exceeded, insufficient_quota) that never succeed on retry. Branch on error.code, not only the status. Overload is 503 server_is_overloaded, not 429 or 529.
- Rate-limit reset headers differ: Anthropic anthropic-ratelimit-*-reset are RFC 3339 timestamps; OpenAI x-ratelimit-reset-* are Go-style durations ('1s', '6m0s'); Gemini documents no reset headers. A shared parser will mis-schedule retries.
- Token accounting differs: Anthropic usage.input_tokens EXCLUDES cached tokens (total = input + cache_creation + cache_read); OpenAI input_tokens/prompt_tokens and Gemini promptTokenCount INCLUDE cached tokens. Gemini candidatesTokenCount EXCLUDES thoughtsTokenCount (thinking is billed as output). Summing fields naively double-counts or under-counts cost.
- OpenAI Responses: 'output_text' is an SDK-only convenience property. Raw HTTP clients must walk output[] -> type=='message' -> content[] type=='output_text' (or 'refusal'). Reasoning items appear before the message item.
- OpenAI: status 'incomplete' with incomplete_details.reason 'max_output_tokens' can come back with zero visible output while still billing reasoning tokens. Treat any status != 'completed' as a failure, and give max_output_tokens a large budget (OpenAI suggests >=25k).
- OpenAI strict mode requires every property in 'required'. Optional fields must be ['T','null'] unions, so the model may return null rather than omit a field; downstream code must treat null as 'unknown', not zero. Anthropic instead reorders output (required fields first) and caps optional params at 24 per request.
- OpenAI 'store' defaults to true (responses kept for at least 30 days). Set store:false for proprietary research prompts unless you need previous_response_id.
- OpenAI 'truncation':'auto' silently drops the earliest conversation items when context overflows. Keep the default 'disabled' so overflow returns 400 instead of silently losing context.
- Gemini structured output has no strict flag, and the docs say 'The model ignores unsupported properties'. A schema keyword the API does not support (e.g. pattern, minLength) is silently dropped rather than rejected, so always validate with jsonschema client-side.
- Gemini: when promptFeedback.blockReason is set, 'candidates' is absent, so response['candidates'][0] raises KeyError. Also check candidates[0].finishReason == 'STOP' (MAX_TOKENS/SAFETY/RECITATION produce partial or empty JSON).
- Gemini reference docs mark responseSchema and _responseJsonSchema deprecated in favor of generationConfig.responseFormat.text {mimeType, schema}. Many official REST examples still use shut-down models (gemini-2.0-flash, gemini-2.5-flash-lite) and the old fields; don't copy the model IDs.
- Gemini API keys can be sent as ?key= in the URL (shown in docs), which leaks them into proxy/access logs and exception traces. Use the x-goog-api-key header.
- Gemini rate limits are per Google Cloud project (not per key), and RPD resets at midnight US Pacific, not UTC. Multiple keys in one project share quota.
- Prompt-cache minimums are silent: below 512 (Opus 5.5/Fable 5.1), 1,024 (Sonnet 5, OpenAI GPT-5.6+), 4,096 (Haiku 4.5, Gemini 3.x) tokens, nothing is cached and no error is raised. Monitor the cached-token fields. Any volatile content (timestamps, as-of dates) placed before the breakpoint also defeats caching.
- Pricing cliffs: OpenAI prompts >272K input tokens bill 2x input / 1.5x output for the WHOLE request. Gemini 3.8 Flash introductory pricing ($0.75/$3.75) doubles on 2027-01-01. OpenAI GPT-5.6+ now charges 1.25x for cache writes.
- claude-haiku-4-5-20251001 retirement is 'not sooner than 2026-10-15' (about 3 weeks away). Hard-coding it as the cheap tier without a fallback model ID risks 404 not_found_error after retirement.
- The bundled claude-api skill cache (dated 2026-06-24) labels claude-opus-5-5 'launching, use only when named'. Live docs (2026-09-24) show it released 2026-09-22 and recommended as the default starting model. Trust live docs.

### UNKNOWN (not verifiable, so the code must fail safe)

- Whether Gemini generateContent 429/503 responses carry a Retry-After header or a google.rpc.RetryInfo retryDelay in error.details: not documented on the pages read. Treat as UNKNOWN and fall back to exponential backoff with jitter.
- Whether Gemini 3.8 Flash returns HTTP 400 or silently ignores temperature/topP/topK: the migration checklist only says to strip them.
- Whether OpenAI returns 400 or silently ignores temperature/top_p when reasoning effort != 'none': the guide only says to remove them.
- Knowledge/training cutoff for gemini-3.8-flash and gemini-3.5-flash-lite: not shown on the model pages fetched.
- Exact accepted string for Gemini responseFormat.text.mimeType: the reference enum says APPLICATION_JSON while the REST examples use 'application/json'. Both appear in official docs; test before relying on one.
- Whether Gemini structured output uses guaranteed constrained decoding comparable to Anthropic/OpenAI strict mode. The docs claim syntactically valid JSON matching the schema, but also say unsupported properties are ignored and give no strict flag.
- Default value of OpenAI text.format.strict / json_schema.strict when omitted (typed 'optional boolean or null'). Always send strict:true explicitly.
- Whether OpenAI model IDs like gpt-6-astra are immutable pinned snapshots or moving aliases: model pages list only 'Default snapshot: gpt-6-astra', with no dated snapshot.
- Semantics of the Anthropic 'x-should-retry' response header (observed live as 'false' on a 401). It is not described on the errors page read.
- Per-tier numeric rate limits (RPM/ITPM/OTPM, TPM, RPD) for any provider: not extracted; they depend on account tier.
- Gemini Interactions API path: docs show both /v1beta/interactions and /v1beta2/interactions (inconsistent). Not relevant if generateContent is used.
- Whether claude-haiku-4-5-20251001 will actually be retired on 2026-10-15. Docs only give 'not sooner than'.

## Research methods & formulas

_Formulas and standard parameters from the academic and quant literature for research-valid backtest tooling: PSR/DSR, purged CV, SUE/EAR, momentum and reversal variants, anomaly decay, spread estimators, block bootstrap and multiple-testing hurdles_

1. **Probabilistic Sharpe Ratio (PSR) formula, Bailey & Lopez de Prado (2012, Journal of Risk 15(2), 'The Sharpe Ratio Efficient Frontier', SSRN 1821643). Checked against the authors' own Python code in Appendix A.3.** [verified]  
   PSR(SR*) = Phi( (SR_hat - SR*) * sqrt(T-1) / sqrt(1 - g3*SR_hat + ((g4-1)/4)*SR_hat^2) ). SR_hat and SR* are per-observation (NOT annualized) Sharpe ratios. T is the number of return observations, g3 is skewness and g4 is RAW kurtosis (Normal = 3). Authors' code: norm.cdf((sr-sr_ref)*(obs-1)**0.5/(1-sr*skew+sr**2*(kurt-1)/4.)**0.5). I re-ran the paper's example (mean=2, sd=sqrt(12), skew=-0.72, kurt=5.78, sr_ref=1/sqrt(12), obs=59.895) and got PSR=0.95000, which matches the paper.  
   Source: https://www.davidhbailey.com/dhbpapers/sharpe-frontier.pdf

2. **Minimum Track Record Length (MinTRL), from the same paper and code** [verified]  
   MinTRL = 1 + (1 - g3*SR_hat + ((g4-1)/4)*SR_hat^2) * (Phi^-1(p) / (SR_hat - SR*))^2, measured in observations, not years. The paper's example reproduces to 59.895 monthly observations (about 4.99 years) at p=0.95. The paper notes MinTRL is in observations, so a weekly and a monthly strategy with the same annualized SR need different track-record lengths.  
   Source: https://www.davidhbailey.com/dhbpapers/sharpe-frontier.pdf

3. **Deflated Sharpe Ratio (DSR) and expected maximum Sharpe under N independent trials, Bailey & Lopez de Prado (2014), JPM 40(5), SSRN 2460551** [verified]  
   DSR = PSR(SR0), i.e. Phi((SR_hat - SR0)*sqrt(T-1)/sqrt(1 - g3*SR_hat + ((g4-1)/4)*SR_hat^2)). SR0 = sqrt(V[{SR_n}]) * ((1-gamma)*Phi^-1(1 - 1/N) + gamma*Phi^-1(1 - 1/(N*e))), where gamma = 0.5772156649 (Euler-Mascheroni), e is Euler's number, V[{SR_n}] is the cross-sectional variance of the trial Sharpe estimates and N is the number of INDEPENDENT trials. Authors' code (Snippet 1): maxZ=(1-emc)*ss.norm.ppf(1-1./numTrials)+emc*ss.norm.ppf(1-1./(numTrials*np.e)); return mu+sigma*maxZ. For DSR the null sets mu = E[{SR_n}] = 0.  
   Source: https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

4. **Reproducing the DSR paper's numerical example confirms the units: raw kurtosis and per-observation SR** [verified]  
   Inputs: N=100, V[SR_n]=1/2 (annualized, so divide by 250 for daily), T=1250 daily obs, SR=2.5 annualized (2.5/sqrt(250) per day), g3=-3, g4=10. My re-implementation gives SR0 = 0.1132 per day and DSR = 0.9004 (paper: about 90%). With N=46, DSR = 0.9505; with Normal returns (g3=0, g4=3) and N=88, DSR = 0.9505. All three match the paper's text.  
   Source: https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

5. **Adjusting DSR's N for correlated trials (Appendix A.3)** [UNVERIFIED]  
   If M trials are correlated, use an implied number of independent trials N_hat. The paper interpolates between average correlation rho_bar -> 1 (N -> 1) and rho_bar -> 0 (N -> M), i.e. N_hat = rho_bar + (1 - rho_bar)*M (eq. 9). The glyphs of eq. 9 were lost in PDF extraction, so this form is rebuilt from the stated endpoints. The paper warns that average correlation is unreliable when M > T (an ill-conditioned correlation matrix) and suggests dimension reduction or clustering, or an information-theoretic (entropy) approach.  
   Source: https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

6. **Purging rule, Lopez de Prado, Advances in Financial ML (Wiley 2018), ch. 7, Snippet 7.1 getTrainTimes (from the publisher's official audiobook companion PDF of the book's snippets)** [verified]  
   Each observation i has a label interval [t0_i = t1.index, t1_i = t1.value]. For each test interval [i, j], drop training observations where (a) i <= t0_i <= j (train starts within test), OR (b) i <= t1_i <= j (train ends within test), OR (c) t0_i <= i AND t1_i >= j (train envelops test).  
   Source: https://gildan-bonus-content.s3.amazonaws.com/GIL2476_AdvancesFinancial/GIL2476_AdvancesFinancial_BonusPDF.pdf

7. **Embargo and the PurgedKFold class (AFML Snippets 7.2-7.3)** [verified]  
   Embargo step = int(T * pctEmbargo) bars and applies only AFTER each test fold. The test end time is extended to times[idx+step] before purging. In PurgedKFold.split (shuffle=False; test folds are contiguous array_split blocks): t0 = t1.index[i]; maxT1Idx = t1.index.searchsorted(t1[test_indices].max()); train = obs with t1 <= t0 (left side), plus indices[maxT1Idx + mbrg:] (right side, after the embargo). The constructor raises if t1 is not a pd.Series, and split raises if X.index != t1.index.  
   Source: https://gildan-bonus-content.s3.amazonaws.com/GIL2476_AdvancesFinancial/GIL2476_AdvancesFinancial_BonusPDF.pdf

8. **Recommended embargo size is about 1% of the sample (h ~ 0.01T)** [UNVERIFIED]  
   Only a secondary source (a thesis summary) confirmed this; the AFML chapter text itself was not accessible, only its code snippets.  
   Source: https://epub.ub.uni-muenchen.de/69183/1/MA_BergerThomas.pdf

9. **Standardized Unexpected Earnings (SUE) as defined in Bernard & Thomas (1990), JAE 13(4):305-340 (read from the Deep Blue PDF text)** [verified]  
   SUE numerator = actual quarterly earnings minus a seasonal-random-walk-with-trend forecast, E[Q_t] = Q_{t-4} + drift. The trend is estimated using up to 36 quarters of history if available, so the numerator is the detrended seasonal difference. Denominator = standard deviation of that forecast error over the trend-estimation period. Deciles are formed from the SUE distribution within each calendar quarter. Abnormal return = firm return minus the NYSE-AMEX same-size-decile return (January 1 market value). Three-day [-2,0] abnormal returns around the next announcements for a long decile 10 / short decile 1 position: +1.32% (t+1), +0.70% (t+2), +0.04% (t+3), negative at t+4 (-0.66% per the Brown et al. summary).  
   Source: https://deepblue.lib.umich.edu/handle/2027.42/28288

10. **SUE with an 8-quarter drift and 8-quarter scaling: Brandt, Kishore, Santa-Clara & Venkatachalam (2008), 'Earnings Announcements are Full of Surprises', SSRN 909563** [verified]  
   SUE_iq = (X_iq - E[X_iq]) / sigma_iq. E[X_iq] = X_{i,q-4} + mu_iq, with mu_iq = (1/8) * sum_{n=1..8} (X_{i,q-n} - X_{i,q-n-4}). sigma_iq = standard deviation of earnings surprises over the last eight quarters. X = Compustat quarterly Data 8. Announcement date = Compustat RDQE. Sample 1987-2004, excluding financials (NAICS 52), utilities (NAICS 22) and stocks under $5 on the trading day before the announcement.  
   Source: https://www.anderson.ucla.edu/documents/areas/fac/finance/santa_clara_surprises.pdf

11. **Earnings Announcement Return (EAR) definition (BKSV 2008, eq. 4)** [verified]  
   EAR_iq = prod_{j=t-1..t+1}(1 + R_ij) - prod_{j=t-1..t+1}(1 + FF_j): the 3-day buy-and-hold return centred on announcement day t, minus the return of the matching Fama-French benchmark portfolio (2 size x 3 book-to-market) that the stock belongs to.  
   Source: https://www.anderson.ucla.edu/documents/areas/fac/finance/santa_clara_surprises.pdf

12. **BKSV point-in-time rules: breakpoints come from the PRIOR quarter, and returns start at t+2** [verified]  
   Quintile breakpoints for SUE and EAR are computed from quarter q-1 observations, not quarter q, explicitly to avoid look-ahead: you cannot know the quarter-q extremes until every firm has announced. Returns cumulate from t+2 through t+n trading days, i.e. the day after the 3-day window ends.  
   Source: https://www.anderson.ucla.edu/documents/areas/fac/finance/santa_clara_surprises.pdf

13. **BKSV drift magnitudes (quintile 5 minus quintile 1)** [verified]  
   SUE spread: 3.23% over 60 trading days, 5.10% over 120, and 6.18% over 240; it weakens and shows signs of reversal after 180 days. EAR spread: 3.27% (60 days), 4.07% (120), 7.55% (240), with no reversal after 180 days. The abstract gives EAR 7.55%/yr, 1.3% more than SUE, and a combined EAR+SUE strategy at about 12.5%/yr. EAR and SUE are largely independent signals.  
   Source: https://www.anderson.ucla.edu/documents/areas/fac/finance/santa_clara_surprises.pdf

14. **Chan, Jegadeesh & Lakonishok (1996), 'Momentum Strategies', JF 51(5):1681-1713: definitions of SUE, ABR and REV6** [verified]  
   SUE = (e_iq - e_{i,q-4}) / sigma_it, where sigma = std dev of (e_q - e_{q-4}) over the preceding eight quarters. This is a seasonal random walk WITHOUT drift, using EPS most recently announced as of month t. ABR = cumulative return minus the EQUALLY-weighted market return from day -2 to day +1 around the most recent announcement. REV6 = 6-month moving average of monthly changes in the I/B/E/S mean FY1 estimate, each scaled by the prior month's price. Decile breakpoints use NYSE stocks only. Returns are measured after skipping the first 5 days post-formation (bid-ask bounce). Sample Jan 1977-Dec 1993. SUE decile 10-1: 6.8% in the first 6 months and 7.5% after a year. ABR: 5.9% (6 months) and 8.3% (1 year).  
   Source: http://www-stat.wharton.upenn.edu/~steele/Courses/434/434Context/Momentum/MomentumStrategiesJF96.pdf

15. **Magnitudes reported for Foster-Olsen-Shevlin (1984) and Bernard-Thomas (1989), both from secondary sources** [UNVERIFIED]  
   B&T 1989's opening (as reproduced by Semantic Scholar and search snippets): FOS 1984 found that a long top-SUE-decile / short bottom-decile position over the 60 trading days after an announcement earns about 25% annualized abnormal return before transaction costs. Brown et al. (NYU WP, 1995) report B&T 1989 as +4.19% over 60 days and 7.74% after 180 days, with about one sixth of it in the first 5 days.  
   Source: https://www.semanticscholar.org/paper/POST-EARNINGS-ANNOUNCEMENT-DRIFT-DELAYED-PRICE-OR-Bernard-Thomas/01354e373f23983ac962c8b133e668332ec26da9

16. **Jegadeesh & Titman (1993), JF 48(1):65-91: the skip period is ONE WEEK, not one month** [verified]  
   J and K each take values of 1, 2, 3 or 4 quarters (3/6/9/12 months), giving 16 strategies, plus 16 more that skip ONE WEEK between formation and holding. Portfolios overlap (each month the position formed K months ago is closed). Deciles are equal-weighted and rebalanced monthly; CRSP 1965-1989. 12/3 strategy: 1.31%/month with no gap and 1.49%/month with a 1-week gap. The 6-month formation period earns about 1%/month for any holding period.  
   Source: https://www.bauer.uh.edu/rsusmel/phd/jegadeesh-titman93.pdf

17. **Standard '12-1' momentum: the Fama-French Mom (UMD) factor construction, read from Ken French's data library** [verified]  
   Ranking uses prior (2-12) month returns. Breakpoints are the 30th and 70th NYSE percentiles; two size groups split at the NYSE median ME; six value-weighted portfolios rebalanced monthly. Mom = 1/2(Small High + Big High) - 1/2(Small Low + Big Low). Eligibility: a price at the end of month t-13, a good return for t-2, and ME at the end of t-1. Missing returns from t-12 to t-3 are allowed (coded -99). Universe: NYSE, AMEX and NASDAQ.  
   Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor.html

18. **Daniel & Moskowitz (2016), 'Momentum crashes', JFE 122(2):221-247: formation rules** [verified]  
   Rank on cumulative return from t-12 to t-2, with a one-month gap to avoid the Jegadeesh (1990) / Lehmann (1990) reversal. Requires at least 8 monthly returns in the 11-month window. Common shares only (CRSP shrcd 10/11) from NYSE, Amex and Nasdaq, with a valid price and shares at formation. 10 value-weighted deciles; membership is fixed within the month except on delisting. Returns are close to close.  
   Source: https://www.kentdaniel.net/papers/published/jfe_16.pdf

19. **Daniel & Moskowitz (2016): dynamic momentum weights, and the in-sample vs out-of-sample distinction** [verified]  
   w*_{t-1} = (1/(2*lambda)) * mu_{t-1} / sigma^2_{t-1}. mu comes from the regression R_WML,t = g0 + g_int * I_B,t-1 * sigma^2_m,t-1 + e. I_B = 1 if the cumulative CRSP VW index return over the past 24 months is negative. sigma^2_m = variance of daily market returns over the prior 126 days. sigma_{t-1} is a linear combination of a GJR-GARCH forecast and the realized standard deviation of the prior 126 daily WML returns. lambda is set for 19% in-sample annual volatility. Sharpe ratios 1934-2013: WML 0.682, constant-vol 1.041, variance-scaled 1.126, dynamic out-of-sample (expanding-window regression through t-1) 1.194, dynamic in-sample (full-sample parameters) 1.202. The dynamic weight can be negative (82 months) and reached 5.37.  
   Source: https://www.kentdaniel.net/papers/published/jfe_16.pdf

20. **Barroso & Santa-Clara (2015), 'Momentum has its moments', JFE 116(1):111-120: volatility-managed momentum formula (read from the working-paper version)** [verified]  
   sigma_hat^2_t = 21 * sum_{j=0..125} r^2_{WML, d_{t-1} - j} / 126. This uses the 126 daily WML returns ending on the last trading day of month t-1, and is NOT demeaned. Scaled return r*_WML,t = (sigma_target / sigma_hat_t) * r_WML,t, with sigma_target = 12% annualized. WML = Ken French top minus bottom decile ranked on t-12..t-2. Sharpe rises from 0.53 to 0.97, excess kurtosis falls from 18.24 to 2.68, and skew improves from -2.47 to -0.42. A footnote reports that 1- and 3-month realized variance or EWMA scaling give nearly identical results.  
   Source: http://www.snifferquant.com/gyantal/Incode/papers/Momentum%20Has%20Its%20Moments(scaling%20Momentum%20by%20vol),2014.pdf

21. **Short-term (1-month) reversal: Jegadeesh (1990), JF 45(3):881-898, abstract read on Wiley** [verified]  
   Monthly individual-stock returns have highly significant negative first-order serial correlation, and significant positive serial correlation at longer lags, especially 12 months. Decile portfolios formed on one-step-ahead forecasts: the extreme-decile abnormal-return difference is 2.49% per month over 1934-1987.  
   Source: https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1990.tb05110.x

22. **Weekly reversal: Lehmann (1990), QJE 105(1):1-28 (NBER WP 2533 text)** [verified]  
   Weeks run Wednesday to Tuesday. Contrarian dollar weights are proportional to -(R_i,t - mean cross-sectional R_t), scaled by the inverse of the sum of positive deviations so the portfolio is $1 long / $1 short. The universe is NYSE/AMEX stocks listed in both week t and week t+k. Transaction cost per security per week = tc * |w_it - w_i,t-1|, with tc the one-way cost. Sweeney (1986) ranges cited: 0.05% (floor traders), 0.1-0.2% (money managers), 0.4% (discount-broker investors). The abstract says profits persist after bid-ask and plausible cost corrections.  
   Source: https://www.nber.org/system/files/working_papers/w2533/w2533.pdf

23. **52-week-high momentum: George & Hwang (2004), JF 59(5):2145-2176 (author-hosted PDF)** [verified]  
   Rank on P_i,t-1 / high_i,t-1, where P is the price at the end of month t-1 and high is the highest price during the 12-month period ending on the last day of month t-1. Winners and losers are the top and bottom 30%, equal-weighted, held 6 months with overlapping (6,6) portfolios. Month-t return = average over the six cohorts formed in t-6..t-1. CRSP 1963-2001. Table I winner-minus-loser: 0.45%/month. The profit rises from 0.45% to 1.23% outside January. Regression tests skip a month; descriptive tables do not. Profits do not reverse in the long run.  
   Source: https://www.bauer.uh.edu/tgeorge/papers/gh4-paper.pdf

24. **Industry momentum: Moskowitz & Grinblatt (1999), JF 54(4):1249-1290** [verified]  
   20 value-weighted industries built from 2-digit SIC codes (NYSE/AMEX/Nasdaq, July 1963-July 1995, about 230 stocks per industry on average). Rank industries on past 6-month return; go equally long the top 3 and short the bottom 3; hold 6 months: 0.43%/month. Skipping a month gives 0.40%/month. Using equal-weighted industry portfolios gives 0.81%/month (10.2%/yr).  
   Source: http://www-stat.wharton.upenn.edu/~steele/Courses/956/Resource/Momentum/MoskowitzGrinblatt99.pdf

25. **Post-publication decay of anomalies: McLean & Pontiff (2016), JF 71(1):5-32 (SSRN and Wiley abstracts)** [verified]  
   Covers 97 cross-sectional predictors. Portfolio returns are 26% lower out-of-sample (after the sample, before publication), which the authors call an upper bound on data mining, and 58% lower after publication. That implies about 32% (58% - 26%) lost to publication-informed trading. Decay is larger for predictors with higher in-sample returns. Returns are higher when concentrated in high-idiosyncratic-risk, low-liquidity stocks.  
   Source: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2156623

26. **Corwin-Schultz (2012) high-low spread estimator, JF 67(2):719-760, taken exactly from the author's official SAS code** [verified]  
   beta = [ln(H_t/L_t)]^2 + [ln(H_{t-1}/L_{t-1})]^2. gamma = [ln(max(H_t,H_{t-1}) / min(L_t,L_{t-1}))]^2. alpha = (sqrt(2*beta) - sqrt(beta))/(3 - 2*sqrt(2)) - sqrt(gamma/(3 - 2*sqrt(2))). S = 2*(exp(alpha) - 1)/(1 + exp(alpha)), the full proportional spread. sigma = (sqrt(beta/2) - sqrt(beta))/(k2*(3-2*sqrt(2))) + sqrt(gamma/(k2^2*(3-2*sqrt(2)))), with k2 = sqrt(8/pi). The primary monthly estimate (MSPREAD_0) is the average of all overlapping two-day estimates in the month, with negative two-day estimates set to 0 first.  
   Source: https://sites.nd.edu/scorwin/files/2024/05/High_Low_Spread_Estimator_SAS_sample.pdf

27. **Corwin-Schultz overnight adjustment and data cleaning (official SAS code)** [verified]  
   Input: daily SPLIT-ADJUSTED high, low and close; PRC = ABS(PRC), because CRSP gives a negative bid/ask midpoint on no-trade days. Days with H=L or non-positive prices get the prior day's retained range, shifted if the close is outside it. Observations with H/L > 8 are dropped. Overnight: if C_{t-1} < L_t, set H_t = H_t - (L_t - C_{t-1}) and L_t = C_{t-1}. If C_{t-1} > H_t, set H_t = C_{t-1} and L_t = L_t + (C_{t-1} - H_t). Lags reset at PERMNO boundaries.  
   Source: https://sites.nd.edu/scorwin/files/2024/05/High_Low_Spread_Estimator_SAS_sample.pdf

28. **Corwin-Schultz calibration against TAQ, 1993-2005 (working-paper text)** [verified]  
   Mean TAQ effective spread 2.60%, median 1.46%. High-low estimator with negatives set to 0: mean 2.65%, median 1.47%. Keeping negatives: mean 1.85%. Omitting negatives: mean 3.66%, and about 20% of stock-months are lost (fewer than 12 observations). These are FULL proportional spreads; a one-way cost is roughly S/2.  
   Source: https://users.nber.org/~confer/2009/mms09/Corwin_Schultz.pdf

29. **Abdi-Ranaldo (2017) close-high-low spread estimator, RFS 30(12):4437-4480 (July 2016 working-paper text served by the AEA)** [verified]  
   Mid-range eta_t = (ln H_t + ln L_t)/2; c_t = ln Close_t. Eq. 9: s^2 = 4*E[(c_t - eta_t)(c_t - eta_{t+1})]. 'Monthly corrected': S = sqrt(max(4*mean_t[(c_t - eta_t)(c_t - eta_{t+1})], 0)). 'Two-day corrected' (the paper's preferred version, eq. 12): s_t = sqrt(max(4*(c_t - eta_t)(c_t - eta_{t+1}), 0)) and S = (1/N)*sum_t s_t. Empirical filter: discard stock-months with fewer than 12 trading days, where a valid day has positive high, low and close and positive volume.  
   Source: https://www.aeaweb.org/conference/2017/preliminary/paper/GbeDTRrB

30. **Trading-cost evidence for anomaly strategies: Novy-Marx & Velikov (2016), RFS 29(1):104-147 (NBER WP 20721)** [verified]  
   Costs use Hasbrouck (2009) Gibbs-sampler effective spreads, which correlate 96.5% with TAQ. These cover small liquidity-demanding trades only; market impact is ignored. Mid-turnover monthly-rebalanced anomalies (14-35% turnover per side per month) cost 20-57 bp/month, often more than half the gross spread. Most anomalies with one-sided monthly turnover under 50% stay significant net of costs when designed to reduce costs; few higher-turnover ones do. A buy/hold spread (sS rule, e.g. enter in the top 10%, exit only when leaving the top 20%) is the most effective single cost-reduction technique.  
   Source: https://www.nber.org/system/files/working_papers/w20721/w20721.pdf

31. **Optimal expected block length for the stationary bootstrap: Politis & White (2004), Econometric Reviews 23(1):53-70, with the Patton-Politis-White (2009) correction, ER 28(4):372-375** [verified]  
   b_opt,SB = (2*G^2 / D_SB)^(1/3) * N^(1/3). G_hat = sum_{k=-M..M} lambda(k/M)*|k|*R_hat(k). Corrected D_SB = 2*g_hat(0)^2, where g_hat(0) = sum_{k=-M..M} lambda(k/M)*R_hat(k); the original 2004 D_SB was wrong. Flat-top window: lambda(t) = 1 for |t| <= 1/2, 2(1-|t|) for 1/2 < |t| <= 1, and 0 otherwise. M = 2*m_hat, where m_hat is the smallest m with |rho_hat(m+k)| < c*sqrt(log10(N)/N) for k = 1..K_N, using c = 2 and K_N = max(5, sqrt(log10 N)). The sqrt glyph in K_N was lost in extraction. The block length grows at rate N^(1/3). The corrected algorithm estimates within 90-110% of the true optimum on average.  
   Source: https://public.econ.duke.edu/~ap172/Patton_Politis_White_2009.pdf

32. **Stationary bootstrap mechanics: Politis & Romano (1994), JASA 89(428):1303-1313** [UNVERIFIED]  
   Resample blocks of random, geometrically distributed length, P(L=m) = (1-p)^(m-1)*p with mean block length 1/p, wrapping circularly, so the pseudo-series is stationary. Seen only in search summaries; the primary paper was not read.  
   Source: https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870

33. **Bonferroni, Holm and BH/BHY procedures as stated in Harvey, Liu & Zhu (2016) (NBER WP 20592 text)** [verified]  
   Bonferroni: reject if p_i <= alpha/M, i.e. p_adj = min(M*p_i, 1). Holm (step-down, controls FWER under any dependence): sort p(1) <= ... <= p(M); let k be the minimum index with p(k) > alpha/(M+1-k); reject H(1)..H(k-1). Adjusted p(i) = min[max_{j<=i}((M-j+1)*p(j)), 1]. BHY (controls FDR): k = maximum index with p(k) <= k*d/(M*c(M)); reject H(1)..H(k). c(M) = sum_{j=1..M} 1/j is valid under ANY dependence (Benjamini-Yekutieli 2001). c(M) = 1 is original Benjamini-Hochberg (1995), valid only under independence or positive dependence.  
   Source: https://www.nber.org/system/files/working_papers/w20592/w20592.pdf

34. **Harvey, Liu & Zhu (2016) t-statistic hurdle, RFS 29(1):5-68** [verified]  
   Abstract: a new factor needs a t-statistic greater than 3.0. The working paper uses 316 published factors and sets FDR d = 1% as the main case. Implied BHY hurdle: about 2.78 at d = 5% (2012). After modelling unpublished tried factors (an estimated 71% missing): Bonferroni 4.01, Holm 3.96, BHY 3.68 (1%) and 3.18 (5%). The authors conclude the minimum threshold is about 3.18.  
   Source: https://academic.oup.com/rfs/article/29/1/5/1843824

35. **Sharpe annualization under serial correlation: Lo (2002), FAJ 58(4):36-52 (seen only in search summaries)** [UNVERIFIED]  
   Monthly Sharpe ratios cannot be annualized by multiplying by sqrt(12) except under special conditions such as IID returns. Serial correlation can overstate a hedge fund's annual Sharpe ratio by up to 65%, and correcting for it can reorder rankings. The exact eta(q) formula was not verified.  
   Source: https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios

### Gotchas

- PSR and DSR use RAW kurtosis (Normal = 3). scipy.stats.kurtosis defaults to fisher=True, which returns EXCESS kurtosis (Normal = 0). Passing it straight in makes (g4-1)/4 negative, shrinks the denominator and inflates PSR/DSR. Use fisher=False or add 3.
- The authors' reference PSR code (sharpe-frontier.pdf) handles '2-moment' and '3-moment' variants by ZEROING the missing moments, including kurtosis (=0, not 3). The Normal case then uses 1 - SR^2/4 instead of the correct 1 + SR^2/2. Do not copy that truncation logic.
- In PSR, DSR and MinTRL, SR_hat, SR* and V[{SR_n}] must all be per-observation (not annualized), and T is the number of observations. Mixing annualized SR with daily T badly overstates significance. The DSR paper's own example divides annualized quantities by sqrt(250) and 250.
- DSR's N means INDEPENDENT trials. Counting only the variants you kept, not every configuration tried, understates SR0 and overstates DSR. Counting M highly correlated variants as independent overstates SR0. Keep a complete trial log.
- The AFML Snippets 7.1-7.4 as printed use APIs that fail on a modern Python 3.14 / pandas 2.x / scikit-learn stack: pd.Series.iteritems and Series.append were removed in pandas 2.0, and BaggingClassifier(base_estimator=...) was renamed to estimator. The snippets also rely on the private sklearn _BaseKFold. Re-implement rather than copy.
- PurgedKFold assumes the t1 Series is sorted, shares X's index and has no NaN label-end times. NaN comparisons evaluate to False, so an unlabeled or open-ended observation would never be purged. The embargo applies only after each test fold, and test folds must be contiguous (shuffle=False).
- SUE ranking look-ahead: Bernard & Thomas (1990) formed deciles from the same calendar quarter's SUE distribution, which is not tradable in real time. BKSV explicitly use quarter q-1 breakpoints for that reason. Use prior-period breakpoints and start returns after the announcement window (t+2 in BKSV; CJL skip 5 days).
- SUE definitions differ between papers and are not interchangeable. B&T 1990 use trend estimated from up to 36 quarters and scaled by the estimation-period std. BKSV use an 8-quarter mean seasonal change as drift and an 8-quarter std. CJL use NO drift and an 8-quarter std of seasonal differences. Pin one definition and version it.
- Announcement timing: EAR and ABR windows ([-1,+1] or [-2,+1]) exist because the announcement date and time-of-day are uncertain; after-close releases move price on day +1. A point-in-time system must know when the announcement timestamp became available and must never trade on day t+1's close using a signal that needs day t+1 data.
- 'Momentum 12-1' is the Fama-French (2-12) or Daniel-Moskowitz (t-12..t-2) convention. Jegadeesh-Titman (1993) skipped only ONE WEEK (or none). Moskowitz-Grinblatt and George-Hwang main tables do not skip; George-Hwang skip a month only in regressions. Mixing conventions changes results.
- Daniel-Moskowitz's headline dynamic momentum strategy uses FULL-SAMPLE GJR-GARCH and regression parameters, which is look-ahead. Use their expanding-window out-of-sample variant (Sharpe 1.194 vs 1.202 in-sample). Dynamic weights can go negative and reach about 5x.
- Barroso-Santa-Clara's variance forecast is 21 * mean of squared daily WML returns over the prior 126 sessions, not demeaned and ending at the last session of month t-1. The 12% target is annual, so convert units consistently. Weights are uncapped leverage multipliers.
- 52-week-high ratio: the high must be split-adjusted (a raw-price high before a split makes the ratio collapse). George-Hwang say 'highest price during the 12-month period'; whether that means daily highs or closes was not confirmed.
- Corwin-Schultz and Abdi-Ranaldo return FULL proportional effective spreads for small trades. A one-way cost is about S/2, and market impact for larger trades is excluded. Averaging choices matter: CS mean 2.65% (negatives set to 0) vs 1.85% (negatives kept) vs 3.66% (negatives dropped).
- CRSP stores a NEGATIVE PRC when there is no trade (bid/ask midpoint), and H/L can equal each other or be missing on zero-volume days. The official CS code takes ABS(PRC), resets bad highs and lows, drops H/L > 8 and applies the overnight adjustment. Skipping these steps biases the estimate silently. Abdi-Ranaldo discard stock-months with fewer than 12 valid trading days.
- Benjamini-Hochberg with c(M)=1 is only guaranteed under independence or positive dependence. Strategy backtests on the same universe are correlated, so use BY's c(M) = sum 1/j, or Holm for FWER. HLZ estimate about 71% of tried factors go unreported, so the M you use should include every configuration tried.
- Do not annualize Sharpe with sqrt(12) or sqrt(252) when returns are autocorrelated (Lo 2002: up to 65% overstatement). Stationary-bootstrap block-length code written before 2009 uses the wrong Politis-White D_SB constant; use the 2009 correction (D_SB = 2*g(0)^2).
- Published anomaly effect sizes should be haircut: McLean-Pontiff find returns 58% lower after publication (26% out-of-sample). Most headline numbers here (PEAD 1974-1986; JT 1965-1989; MG 1963-1995) are pre-decimalization and pre-publication.

### UNKNOWN (not verifiable, so the code must fail safe)

- The exact form of DSR Appendix A.3 eq. (9) for the implied number of independent trials: glyphs were lost in PDF extraction. N_hat = rho_bar + (1-rho_bar)*M is rebuilt from the stated limits.
- The AFML recommended embargo size (h ~ 0.01T) was confirmed only by a secondary source. The book chapter text was not accessible, only its code snippets.
- Foster, Olsen & Shevlin (1984) exact SUE model and estimation window, and the ~25% annualized figure: JSTOR and TAR were not accessible. The Bernard & Thomas (1989) figures of 4.19% (60 days) and 7.74% (180 days) and the -0.66% at t+4 come from secondary summaries.
- Bernard & Thomas (1989) exact SUE estimation window and minimum-quarters requirement. Only the 1990 paper's 'up to 36 quarters' rule was read.
- Politis & Romano (1994) primary text was not read; the geometric block-length description comes from search summaries.
- The K_N term in Politis-White (2004): extraction lost the sqrt symbol. max(5, sqrt(log10 N)) is the standard reading but was not seen verbatim.
- Lo (2002) exact annualization factor eta(q) = q / sqrt(q + 2*sum_{k=1}^{q-1}(q-k)*rho_k) and the IID standard error of SR were NOT verified from the primary paper.
- No primary per-liquidity-bucket half-spread table (by size or ADV decile) for current US equities was extracted. The Novy-Marx-Velikov Figure 1 values and Hasbrouck's Gibbs estimates by size were not read. Current (post-2020) cost levels are unverified.
- Whether George & Hwang (2004) used daily high prices or closing prices for the 52-week high.
- Holm (1979) and Benjamini-Hochberg (1995) original papers were not read directly; the procedures were checked via Harvey-Liu-Zhu's restatement.
- The Barroso & Santa-Clara numbers come from a working-paper copy with third-party annotations; they were not re-checked against the final JFE version.
- Jegadeesh (1990) portfolio-construction details (the regression forecast model) and Lehmann (1990) profit magnitudes were not extracted; only the abstract-level results were confirmed.
