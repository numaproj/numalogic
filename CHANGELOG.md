# Changelog

## v0.11.1 (2024-06-13)

 * [8f6c8ef](https://github.com/numaproj/numalogic/commit/8f6c8efd0cc65194a672b3dabfc483babf97114d) feat: vanilla ae with independent channels  (#392)
 * [1239acd](https://github.com/numaproj/numalogic/commit/1239acd6c951770b44e918f25cc76fd48ce7c526) Multichannel config (#390)
 * [56a6ac1](https://github.com/numaproj/numalogic/commit/56a6ac19378ed5090db101a11418b340e6dfbf89) feat: Support Python 3.12; upgrade pynumaflow to 0.7.1 (#383)
 * [c61f59d](https://github.com/numaproj/numalogic/commit/c61f59d43912186ca07d2d543e1bd06feeb11f06) fix: release new patch version (#385)
 * [4122625](https://github.com/numaproj/numalogic/commit/412262538d339e17097fb215b841db9802184d60) fix: make pivot as optional (#384)
 * [309f16a](https://github.com/numaproj/numalogic/commit/309f16a90774adc47623fed5fbb43e0c81da93e9) Added multichannel autoencoder with test cases (#382)

### Contributors

 * Avik Basu
 * Kushal Batra
 * Miroslav Boussarov

## v0.11.0a0 (2024-05-31)

 * [1b6511b](https://github.com/numaproj/numalogic/commit/1b6511b4986f1ce3d2d569dc35c3dc48e78742bf) feat: Support Python 3.12; upgrade pynumaflow to 0.7.1

### Contributors

 * Avik Basu

## v0.11.0 (2024-06-10)

 * [56a6ac1](https://github.com/numaproj/numalogic/commit/56a6ac19378ed5090db101a11418b340e6dfbf89) feat: Support Python 3.12; upgrade pynumaflow to 0.7.1 (#383)

### Contributors

 * Avik Basu

## v0.10.2 (2024-05-31)

 * [c61f59d](https://github.com/numaproj/numalogic/commit/c61f59d43912186ca07d2d543e1bd06feeb11f06) fix: release new patch version (#385)
 * [fb662db](https://github.com/numaproj/numalogic/commit/fb662db89f9e5d2ab31dd7a24bce960ce5151dcf) fix: release new patch version

### Contributors

 * Kushal Batra

## v0.10.1 (2024-05-31)

 * [4122625](https://github.com/numaproj/numalogic/commit/412262538d339e17097fb215b841db9802184d60) fix: make pivot as optional (#384)
 * [309f16a](https://github.com/numaproj/numalogic/commit/309f16a90774adc47623fed5fbb43e0c81da93e9) Added multichannel autoencoder with test cases (#382)
 * [934d8cf](https://github.com/numaproj/numalogic/commit/934d8cfe28a964915545ffa262ba162c9a85612f) Feat/vanilla ae refactor (#379)

### Contributors

 * Avik Basu
 * Kushal Batra
 * Miroslav Boussarov

## v0.10.0a0 (2024-05-08)

 * [6f46250](https://github.com/numaproj/numalogic/commit/6f46250fae787913f01d6fe8076219a7bdbf100a) fix: filters
 * [87c16da](https://github.com/numaproj/numalogic/commit/87c16dab543d7e0d28aabe2e7d2b094a4aa5e2da) fix: filters
 * [bbcfd47](https://github.com/numaproj/numalogic/commit/bbcfd47c9ad7a0eba8f89428bad7b697999b199c) fix: threshold and filters
 * [984650b](https://github.com/numaproj/numalogic/commit/984650bd3a0063cb51f1b9dfc6e1d700c6591170) feat: agg from conf for multi pivot (#378)
 * [402aabf](https://github.com/numaproj/numalogic/commit/402aabfe19f3857e9b467e0edbfd7da2160842f2) Fix: Changing pivot to pivot_table to support aggregation (#376)
 * [141e9a0](https://github.com/numaproj/numalogic/commit/141e9a04cf526d81a85064eac6029cecf0ad9c2f) Postproc support for None (#375)
 * [e1ae3ee](https://github.com/numaproj/numalogic/commit/e1ae3ee2d736993d659b2c7b7112effddc340b2c) feat: multi column pivot for druid connector (#374)
 * [98c5766](https://github.com/numaproj/numalogic/commit/98c5766edb10087f3b50224c3a5be551040bb829) Static filter (#373)
 * [f9ebfaf](https://github.com/numaproj/numalogic/commit/f9ebfaf339b7ebb856277342554e2f1b19f0d0ff) fix: add threshold in metadata
 * [cbf4dbd](https://github.com/numaproj/numalogic/commit/cbf4dbd1183b2412e3f248e7b06c09606639def3) try: replicate og vanilla ae
 * [8b2e939](https://github.com/numaproj/numalogic/commit/8b2e9399a78179a741b2d4266e23868aab6dad3b) feat: enable feature transforms
 * [eed9ad4](https://github.com/numaproj/numalogic/commit/eed9ad4f430e26e73ce86d11e4f4571d761ba3af) fix: adjust factor
 * [9ad6d8d](https://github.com/numaproj/numalogic/commit/9ad6d8ddd168093a771a41a13432867efff99360) feat: fallback to stddev if threshold is too low
 * [5593be3](https://github.com/numaproj/numalogic/commit/5593be3882932329d58095ade6bd4d439195ef32) fix: C contiguous error for exp mov avg tx
 * [8321914](https://github.com/numaproj/numalogic/commit/832191478e654191f8bf8143fb0e97e2ab0d6d93) correct struct log for ack_insuf_data
 * [b36356c](https://github.com/numaproj/numalogic/commit/b36356c656285981ac8dc12f96586182d9404906) fix: percentile scaler, exp mov avg, sigmoid norm (#369)
 * [f7bd30b](https://github.com/numaproj/numalogic/commit/f7bd30b0c838738d7ee8321436d7707dcb7623a9) feat: add PercentileScaler

### Contributors

 * Avik Basu
 * Gulshan Bhatia
 * Kushal Batra
 * Nandita Koppisetty
 * s0nicboOm

## v0.10.0 (2024-05-14)

 * [934d8cf](https://github.com/numaproj/numalogic/commit/934d8cfe28a964915545ffa262ba162c9a85612f) Feat/vanilla ae refactor (#379)
 * [f29f771](https://github.com/numaproj/numalogic/commit/f29f77174e59131fcf5ffad899b5123e31c288e6) Adding RDS Trainer UDF changes (#371)

### Contributors

 * Avik Basu
 * Saisharath Reddy Kondakindi

## v0.9.2 (2024-05-06)

 * [8bba9ca](https://github.com/numaproj/numalogic/commit/8bba9caf7b6f0f5dcda3200e8941e0b044586749) Optimize UDF logs (#370)

### Contributors

 * Gulshan Bhatia

## v0.9.1a11 (2024-05-07)

 * [984650b](https://github.com/numaproj/numalogic/commit/984650bd3a0063cb51f1b9dfc6e1d700c6591170) feat: agg from conf for multi pivot (#378)

### Contributors

 * Nandita Koppisetty

## v0.9.1a10 (2024-05-03)

 * [402aabf](https://github.com/numaproj/numalogic/commit/402aabfe19f3857e9b467e0edbfd7da2160842f2) Fix: Changing pivot to pivot_table to support aggregation (#376)

### Contributors

 * Nandita Koppisetty

## v0.9.1a9 (2024-05-03)

 * [141e9a0](https://github.com/numaproj/numalogic/commit/141e9a04cf526d81a85064eac6029cecf0ad9c2f) Postproc support for None (#375)
 * [e1ae3ee](https://github.com/numaproj/numalogic/commit/e1ae3ee2d736993d659b2c7b7112effddc340b2c) feat: multi column pivot for druid connector (#374)

### Contributors

 * Kushal Batra
 * Nandita Koppisetty

## v0.9.1a8 (2024-05-01)

 * [98c5766](https://github.com/numaproj/numalogic/commit/98c5766edb10087f3b50224c3a5be551040bb829) Static filter (#373)

### Contributors

 * Kushal Batra

## v0.9.1a7 (2024-04-29)

 * [f9ebfaf](https://github.com/numaproj/numalogic/commit/f9ebfaf339b7ebb856277342554e2f1b19f0d0ff) fix: add threshold in metadata

### Contributors

 * s0nicboOm

## v0.9.1a6 (2024-04-24)

 * [cbf4dbd](https://github.com/numaproj/numalogic/commit/cbf4dbd1183b2412e3f248e7b06c09606639def3) try: replicate og vanilla ae
 * [8b2e939](https://github.com/numaproj/numalogic/commit/8b2e9399a78179a741b2d4266e23868aab6dad3b) feat: enable feature transforms

### Contributors

 * Avik Basu

## v0.9.1a5 (2024-04-23)

 * [eed9ad4](https://github.com/numaproj/numalogic/commit/eed9ad4f430e26e73ce86d11e4f4571d761ba3af) fix: adjust factor

### Contributors

 * Avik Basu

## v0.9.1a4 (2024-04-23)

 * [9ad6d8d](https://github.com/numaproj/numalogic/commit/9ad6d8ddd168093a771a41a13432867efff99360) feat: fallback to stddev if threshold is too low

### Contributors

 * Avik Basu

## v0.9.1a3 (2024-04-22)

 * [5593be3](https://github.com/numaproj/numalogic/commit/5593be3882932329d58095ade6bd4d439195ef32) fix: C contiguous error for exp mov avg tx

### Contributors

 * Avik Basu

## v0.9.1a2 (2024-04-22)

 * [8321914](https://github.com/numaproj/numalogic/commit/832191478e654191f8bf8143fb0e97e2ab0d6d93) correct struct log for ack_insuf_data

### Contributors

 * Gulshan Bhatia

## v0.9.1a1 (2024-04-22)

 * [b36356c](https://github.com/numaproj/numalogic/commit/b36356c656285981ac8dc12f96586182d9404906) fix: percentile scaler, exp mov avg, sigmoid norm (#369)

### Contributors

 * Avik Basu

## v0.9.1a0 (2024-04-18)

 * [f7bd30b](https://github.com/numaproj/numalogic/commit/f7bd30b0c838738d7ee8321436d7707dcb7623a9) feat: add PercentileScaler
 * [6975394](https://github.com/numaproj/numalogic/commit/697539487d21549f03052de6f53082ac90259865) logging fix (#367)

### Contributors

 * Avik Basu
 * Gulshan Bhatia

## v0.9.1 (2024-04-17)

 * [c3991bb](https://github.com/numaproj/numalogic/commit/c3991bb59544d2189b787a3a659721f98022dbf9) Refactor logging in UDFs (#356)
 * [be29f02](https://github.com/numaproj/numalogic/commit/be29f027c9c4b5a275fd51845fa847862a280b5c) Aws connectors (#358)

### Contributors

 * Gulshan Bhatia
 * Saisharath Reddy Kondakindi

## v0.9.0 (2024-04-17)

 * [ad20add](https://github.com/numaproj/numalogic/commit/ad20addff7789b1e03ecc1c17469dc949747b54b) fix: remove print (#363)
 * [61f6598](https://github.com/numaproj/numalogic/commit/61f6598f39132a8bb4ee1b1844d166040483383d) feat: initial support for flattened vector in backtest (#361)
 * [0cbaa52](https://github.com/numaproj/numalogic/commit/0cbaa52b43d3175564dc97dfdfbf0c17cfaa4f1a) feat: introduce stride in dataset (#360)
 * [db2cc4f](https://github.com/numaproj/numalogic/commit/db2cc4fad4960e9a9e9bd289d23a54d918ff8038) feat: support for train transformers (#354)
 * [377ada2](https://github.com/numaproj/numalogic/commit/377ada22a4fdec75b6d6dcb7bc11dbdddcccbfa1) feat: generate static scores for backtesting  (#355)

### Contributors

 * Avik Basu
 * Kushal Batra

## v0.8.0 (2024-03-05)

 * [8dee66b](https://github.com/numaproj/numalogic/commit/8dee66bc29e9799c39da656a3beed26c28242c99) feat!: support static threshold score adjustment (#350)

### Contributors

 * Avik Basu

## v0.8.dev0 (2024-02-28)

 * [4e93088](https://github.com/numaproj/numalogic/commit/4e93088c6694457c431fb32e20d5d4477c4579cf) bump dev version
 * [81971dc](https://github.com/numaproj/numalogic/commit/81971dcf5045f7e6350067ce73ee84609b7b810a) fix: static threshold
 * [db943a6](https://github.com/numaproj/numalogic/commit/db943a6768136bd61bad1cb6614a68680c10f1ec) fix: conditional fwd tags
 * [c0225a9](https://github.com/numaproj/numalogic/commit/c0225a967b08e360c426fe35f886c129f2a50363) refactor
 * [8f8a532](https://github.com/numaproj/numalogic/commit/8f8a532c3312048489834b570e25c7b8773ebd20) feat: score adjustment with joined trainer vtx
 * [ac0eda1](https://github.com/numaproj/numalogic/commit/ac0eda176187c41e4b1aa960b8d0d1cfaeacda38) more tests
 * [6623e2b](https://github.com/numaproj/numalogic/commit/6623e2b1d0f85180c472037809b01b4176549eeb) feat: trainer join vertex for preprocess & inference
 * [fb97b3a](https://github.com/numaproj/numalogic/commit/fb97b3a5bbe2668bb19a6680c04a6f337f3b05fa) feat: allow chunked druid fetch (#349)
 * [d8bda0e](https://github.com/numaproj/numalogic/commit/d8bda0e159cc89af95daa70b2acd6b276a4f49c1) feat: Multivariate Anomaly Detection (#343)
 * [d0dcb47](https://github.com/numaproj/numalogic/commit/d0dcb472fe5fcb80df33c51088332f946180e55a) feat: score adjuster using static threshold

### Contributors

 * Avik Basu

## v0.7.0 (2024-02-08)

 * [b41a7f7](https://github.com/numaproj/numalogic/commit/b41a7f72ca92bd635e20d2cb125d45329038448a) fix: comments
 * [780f2e2](https://github.com/numaproj/numalogic/commit/780f2e23cccf3648b385499c589347c3bed0461e) fix merge issues
 * [9de1e83](https://github.com/numaproj/numalogic/commit/9de1e83f133b759431ddbf5ab690cd7d0cfca5d3) fix: backtest
 * [c418b5a](https://github.com/numaproj/numalogic/commit/c418b5aed2807bc1d8fa3f4a943dd360e998b10e) tmpfix: numalogic opex tags for query filter
 * [bc6d08c](https://github.com/numaproj/numalogic/commit/bc6d08c4abf6626a89c2b7555d486c12d24dd301) feat: Score config in postprocess
 * [a913545](https://github.com/numaproj/numalogic/commit/a913545c1d1e2197d2ce65714f675a702dc56226) fix: get item for final score
 * [cc681a3](https://github.com/numaproj/numalogic/commit/cc681a39fc644ffc5d69e717f291dd1a9e304ce7) tmp: testing per feature score
 * [634b2cf](https://github.com/numaproj/numalogic/commit/634b2cf5e3acf1236f30fb4dfdf25578161b5e64) fix: factory class
 * [36db5df](https://github.com/numaproj/numalogic/commit/36db5df3bf5c5bb0ebe567445028f9ddd2ab9c1d) tests: median thresh tests
 * [777b5e2](https://github.com/numaproj/numalogic/commit/777b5e2415b1e3177e780c1d6806b4d14f77cef1) fix: import
 * [a075761](https://github.com/numaproj/numalogic/commit/a075761c4b5cf730ba21f457716a0c5f22cf2d88) feat: add percentile thresholding
 * [fbf171f](https://github.com/numaproj/numalogic/commit/fbf171f3b4b39831a0ee7698ed55ca3425b9398f) fix: pipeline id keyerr in tools.py
 * [add642c](https://github.com/numaproj/numalogic/commit/add642c3b7ab49f8877fa732c2e6e7362bd25df8) fix: pipeline id key error
 * [dc35c80](https://github.com/numaproj/numalogic/commit/dc35c80d26979fd27129405c98557c3aa415106a) try: better docker workflow
 * [faf1e32](https://github.com/numaproj/numalogic/commit/faf1e32da4b4ed99a8b1b70acdc905ab97aeccb2) fix: default ml pipe id in preproc
 * [03faf8f](https://github.com/numaproj/numalogic/commit/03faf8f723556a074703ca914ceaa28f92289f81) fix: vae nsamples
 * [16c02f1](https://github.com/numaproj/numalogic/commit/16c02f158e701a95d1d18e663e83ea19581dd393) feat: add beta parameter for disentanglement
 * [5ec5883](https://github.com/numaproj/numalogic/commit/5ec5883e2a3dcfd1df8f477c3cbd8c2be3459e34) feat: difference transform
 * [46363d3](https://github.com/numaproj/numalogic/commit/46363d3c0ab01d7a6a6c157ffb9beff294850fa5) feat: output threshold + final scores
 * [7f9317c](https://github.com/numaproj/numalogic/commit/7f9317c60209b5f463ecaadb328bb985a7051783) feat: support loading nl conf
 * [3035407](https://github.com/numaproj/numalogic/commit/303540770d7452b864df591495d54ae6858c1f16) feat: multivariate backtesting
 * [c0596d7](https://github.com/numaproj/numalogic/commit/c0596d7828fcface27fe6ce4bfbc5ece47de1035) add more logs
 * [3ca8c15](https://github.com/numaproj/numalogic/commit/3ca8c152a6d35afb69bdbe9dfc872c8de33c3c9a) tmp: try exp mov average
 * [43c1ec8](https://github.com/numaproj/numalogic/commit/43c1ec81ed69eeb6852fd56bf51232d00708cbee) feat: add transforms and robust thresholding
 * [19ddc8e](https://github.com/numaproj/numalogic/commit/19ddc8ead61e1b720d69b8556385fabfb2746369) Demo branch (#335)

### Contributors

 * Avik Basu
 * Kushal Batra

## v0.6.3 (2024-02-08)

 * [4dbde41](https://github.com/numaproj/numalogic/commit/4dbde415ed214fb984fd3989775ee40c9f7b6bf9) Fix: Get metrics from _conf.metrics on Trainer to avoid issue with Flattening the matrix (#346)

### Contributors

 * shashank10456

## v0.6.2 (2024-01-31)

 * [824a55d](https://github.com/numaproj/numalogic/commit/824a55dc7c755dedc11ade4b98a0b1c2c1acd6b1) fix: add FlattenVector transformer (#344)
 * [146ec00](https://github.com/numaproj/numalogic/commit/146ec0090a87c2ded291a229573e215c556ca0f8) fix: add unified conf (#342)
 * [870f263](https://github.com/numaproj/numalogic/commit/870f263166929c904e3c37e7f3ae530a378a963c) fix: take mean before calculating the thresholds (#340)
 * [dbb510f](https://github.com/numaproj/numalogic/commit/dbb510fa0ba8e0696200d4aa62b046612db7527b) fix: add max value map for cliping the value (#339)
 * [80ef431](https://github.com/numaproj/numalogic/commit/80ef43170bc119369aad6661ffa4a8417ba2b063) "Source" tag for metrics (#338)
 * [b292553](https://github.com/numaproj/numalogic/commit/b292553adaa77e6b09731625a9e04849f7651704) Add pl conf (#336)
 * [28fa28f](https://github.com/numaproj/numalogic/commit/28fa28fa972e347548f3b0d56e52605abc29f0d1) Metrics (#322)

### Contributors

 * Kushal Batra

## v0.6.1.dev5 (2023-11-30)

 * [dfc383a](https://github.com/numaproj/numalogic/commit/dfc383a37ee0c950ff01cb9559427f8a13757e60) feat: support both base conf and app conf (#328)
 * [8b7f45f](https://github.com/numaproj/numalogic/commit/8b7f45f01427a2614f66d39a643c2c06d423f217) feat!: support full multivariate prometheus fetching (#325)

### Contributors

 * Avik Basu

## v0.6.1.dev4 (2023-11-21)

 * [fb80cb6](https://github.com/numaproj/numalogic/commit/fb80cb6d09e88750e3373fe710736ccba8de7226) fix: inf filling

### Contributors

 * Avik Basu

## v0.6.1.dev3 (2023-11-21)

 * [5aa4c13](https://github.com/numaproj/numalogic/commit/5aa4c134ac57b6072e3e2e7307259f96ea279464) fix: use ckeys aligning with config in pre, post and inference vtx

### Contributors

 * Avik Basu

## v0.6.1.dev2 (2023-11-20)

 * [b761c94](https://github.com/numaproj/numalogic/commit/b761c948afa44e81418d67a658a33b01234eb874) fix: composite key unzip

### Contributors

 * Avik Basu

## v0.6.1.dev1 (2023-11-20)

 * [9d1b999](https://github.com/numaproj/numalogic/commit/9d1b99944954150bc2efa8bffd07ed78eb1a6dbe) fix: send conf keys instead of request keys to trainer

### Contributors

 * Avik Basu

## v0.6.1.dev0 (2023-11-20)

 * [14e1a66](https://github.com/numaproj/numalogic/commit/14e1a66b190ea01d7a9ab6c9bda1eac7567f7f98) fix: mock method
 * [8369510](https://github.com/numaproj/numalogic/commit/8369510005b90f7f1fa0d0b3ed4a62e625de9924) feat!: separate Prom trainer and Druid trainer
 * [7be6ddc](https://github.com/numaproj/numalogic/commit/7be6ddcbc73308963aef3e835dd46332a0ba25d6) add more tests
 * [c783ba8](https://github.com/numaproj/numalogic/commit/c783ba8ecbc46a859a9d5a075ca1404b55e8f5b6) feat!: support mv query in fetch() method

### Contributors

 * Avik Basu

## v0.6.1.a9 (2024-01-20)

 * [870f263](https://github.com/numaproj/numalogic/commit/870f263166929c904e3c37e7f3ae530a378a963c) fix: take mean before calculating the thresholds (#340)
 * [dbb510f](https://github.com/numaproj/numalogic/commit/dbb510fa0ba8e0696200d4aa62b046612db7527b) fix: add max value map for cliping the value (#339)
 * [80ef431](https://github.com/numaproj/numalogic/commit/80ef43170bc119369aad6661ffa4a8417ba2b063) "Source" tag for metrics (#338)

### Contributors

 * Kushal Batra

## v0.6.1.a7 (2023-12-22)


### Contributors


## v0.6.1 (2024-01-29)

 * [146ec00](https://github.com/numaproj/numalogic/commit/146ec0090a87c2ded291a229573e215c556ca0f8) fix: add unified conf (#342)
 * [870f263](https://github.com/numaproj/numalogic/commit/870f263166929c904e3c37e7f3ae530a378a963c) fix: take mean before calculating the thresholds (#340)
 * [dbb510f](https://github.com/numaproj/numalogic/commit/dbb510fa0ba8e0696200d4aa62b046612db7527b) fix: add max value map for cliping the value (#339)
 * [80ef431](https://github.com/numaproj/numalogic/commit/80ef43170bc119369aad6661ffa4a8417ba2b063) "Source" tag for metrics (#338)
 * [b292553](https://github.com/numaproj/numalogic/commit/b292553adaa77e6b09731625a9e04849f7651704) Add pl conf (#336)
 * [28fa28f](https://github.com/numaproj/numalogic/commit/28fa28fa972e347548f3b0d56e52605abc29f0d1) Metrics (#322)
 * [dfc383a](https://github.com/numaproj/numalogic/commit/dfc383a37ee0c950ff01cb9559427f8a13757e60) feat: support both base conf and app conf (#328)
 * [8b7f45f](https://github.com/numaproj/numalogic/commit/8b7f45f01427a2614f66d39a643c2c06d423f217) feat!: support full multivariate prometheus fetching (#325)
 * [c967f20](https://github.com/numaproj/numalogic/commit/c967f200ddf757d239c65fd97c488023e5daf2ad) feat: support multivar prom (#317)
 * [46cdfcd](https://github.com/numaproj/numalogic/commit/46cdfcdd42624263ff796f09ef3b8f20e8bb36a3) add retrain logic for insufficient data (#321)
 * [0bbb53d](https://github.com/numaproj/numalogic/commit/0bbb53d63d300c20c545a107257bf78603520013) doc: Update to fix examples ipynb files (#318)
 * [61f4575](https://github.com/numaproj/numalogic/commit/61f4575eb941e178d4ad128e870983ff026a78bf) chore!: unify and refactor trainer (#315)

### Contributors

 * Avik Basu
 * Haripriya
 * Kushal Batra

## v0.6.0rc0 (2023-10-12)

 * [9787b2d](https://github.com/numaproj/numalogic/commit/9787b2d9609f7c7743bdf26df79270c0104f035e) update version
 * [a12948f](https://github.com/numaproj/numalogic/commit/a12948f7830fb227909bd64100795bb9c8072217) chore!: unify and refactor trainer
 * [509e38a](https://github.com/numaproj/numalogic/commit/509e38a24fbbbf0a87bfa156809e7874b53d0261) feat: support multivariate threshold output (#314)

### Contributors

 * Avik Basu

## v0.6.0a11 (2023-10-04)

 * [64c2e95](https://github.com/numaproj/numalogic/commit/64c2e95b3b1ba34f3eecdad9906d9857e63af48b) add: druidfetcher support for different configId (#307)

### Contributors

 * Kushal Batra

## v0.6.0a10 (2023-10-02)

 * [f25f49a](https://github.com/numaproj/numalogic/commit/f25f49a3a67ae74092212d77ea609bf77fce81f3) feat: add jitter (#305)

### Contributors

 * Kushal Batra

## v0.6.0a9 (2023-10-02)

 * [d7f9605](https://github.com/numaproj/numalogic/commit/d7f96059fc2d911fc81bbeb02706fcee4ff10331) feat: update druid query context (#304)

### Contributors

 * shrivardhan

## v0.6.0a8 (2023-09-27)

 * [a0e0ad0](https://github.com/numaproj/numalogic/commit/a0e0ad0bc364a1148240b98e911158aa1e971a17) fix: docker extra args error (#302)

### Contributors

 * Avik Basu

## v0.6.0a7 (2023-09-27)

 * [dfab26e](https://github.com/numaproj/numalogic/commit/dfab26e6f12c7c06d016d8062d2a80c31185dd96) refactor druid connector (#301)
 * [2973dd2](https://github.com/numaproj/numalogic/commit/2973dd2385d7cf5411e0bc385c418cb16d4dad56) feat: add dedup logic (#299)

### Contributors

 * Kushal Batra
 * shrivardhan

## v0.6.0a6 (2023-09-26)

 * [fdec237](https://github.com/numaproj/numalogic/commit/fdec237a436809084b038147006003dd39ac6744) fix: trainer bug (#297)

### Contributors

 * Kushal Batra

## v0.6.0a5 (2023-09-22)

 * [d249942](https://github.com/numaproj/numalogic/commit/d249942d393e9d610e90449c20057cd8a3e7ce67) fix druid connector with tests (#296)

### Contributors

 * shrivardhan

## v0.6.0a4 (2023-09-20)

 * [98e376a](https://github.com/numaproj/numalogic/commit/98e376a488aead27eae75ba463f681ff7e5e1192) fix: udf server start error (#294)

### Contributors

 * Avik Basu

## v0.6.0a3 (2023-09-19)

 * [f55312b](https://github.com/numaproj/numalogic/commit/f55312bc2d02c036308b4fd17f91aeca20546986) fix:  pydruid version update (#293)

### Contributors

 * Kushal Batra

## v0.6.0a2 (2023-09-19)

 * [21f85f9](https://github.com/numaproj/numalogic/commit/21f85f97a15e2668162fb9ccc56b42e967407602) try : wfl (#290)

### Contributors

 * Kushal Batra

## v0.6.0a1 (2023-09-19)


### Contributors


## v0.6.0 (2023-11-14)

 * [46cdfcd](https://github.com/numaproj/numalogic/commit/46cdfcdd42624263ff796f09ef3b8f20e8bb36a3) add retrain logic for insufficient data (#321)
 * [0bbb53d](https://github.com/numaproj/numalogic/commit/0bbb53d63d300c20c545a107257bf78603520013) doc: Update to fix examples ipynb files (#318)
 * [61f4575](https://github.com/numaproj/numalogic/commit/61f4575eb941e178d4ad128e870983ff026a78bf) chore!: unify and refactor trainer (#315)
 * [509e38a](https://github.com/numaproj/numalogic/commit/509e38a24fbbbf0a87bfa156809e7874b53d0261) feat: support multivariate threshold output (#314)
 * [64c2e95](https://github.com/numaproj/numalogic/commit/64c2e95b3b1ba34f3eecdad9906d9857e63af48b) add: druidfetcher support for different configId (#307)
 * [f25f49a](https://github.com/numaproj/numalogic/commit/f25f49a3a67ae74092212d77ea609bf77fce81f3) feat: add jitter (#305)
 * [d7f9605](https://github.com/numaproj/numalogic/commit/d7f96059fc2d911fc81bbeb02706fcee4ff10331) feat: update druid query context (#304)
 * [a0e0ad0](https://github.com/numaproj/numalogic/commit/a0e0ad0bc364a1148240b98e911158aa1e971a17) fix: docker extra args error (#302)
 * [dfab26e](https://github.com/numaproj/numalogic/commit/dfab26e6f12c7c06d016d8062d2a80c31185dd96) refactor druid connector (#301)
 * [2973dd2](https://github.com/numaproj/numalogic/commit/2973dd2385d7cf5411e0bc385c418cb16d4dad56) feat: add dedup logic (#299)
 * [fdec237](https://github.com/numaproj/numalogic/commit/fdec237a436809084b038147006003dd39ac6744) fix: trainer bug (#297)
 * [d249942](https://github.com/numaproj/numalogic/commit/d249942d393e9d610e90449c20057cd8a3e7ce67) fix druid connector with tests (#296)
 * [98e376a](https://github.com/numaproj/numalogic/commit/98e376a488aead27eae75ba463f681ff7e5e1192) fix: udf server start error (#294)
 * [f55312b](https://github.com/numaproj/numalogic/commit/f55312bc2d02c036308b4fd17f91aeca20546986) fix:  pydruid version update (#293)
 * [21f85f9](https://github.com/numaproj/numalogic/commit/21f85f97a15e2668162fb9ccc56b42e967407602) try : wfl (#290)
 * [3e89ccd](https://github.com/numaproj/numalogic/commit/3e89ccd3ac5e78a26eef735daa93a3aedf4a0668) fix: allow full df scores in backtest (#288)
 * [bc1c627](https://github.com/numaproj/numalogic/commit/bc1c627cf89e9a41fa2194df3a2a1ed33e5d989e) feat: add multiple save for redis registry (#281)
 * [a364721](https://github.com/numaproj/numalogic/commit/a364721d617f42a56badde501b98e4e2684ea684) feat: initial version of backtest tool (#282)
 * [0cdc257](https://github.com/numaproj/numalogic/commit/0cdc257e11c1c92fedc3409435091141b1a8c71e) feat: use well-defined dimensions instead of strings (#284)
 * [de8930a](https://github.com/numaproj/numalogic/commit/de8930a38b5069adb53dbdf58bad3d68f2b79ce9) feat!: numalogic udfs (#271)
 * [c62c902](https://github.com/numaproj/numalogic/commit/c62c9023c5e0f37aefa4a6d753ae52dae4b626c1) doc: update quick-start.md (#246)
 * [52a65c0](https://github.com/numaproj/numalogic/commit/52a65c0b05df53fc5f6497bf0c71991482d88be5) fix: improve metadata serialization (#244)
 * [8482c01](https://github.com/numaproj/numalogic/commit/8482c0168d89faf7d7d1cb19a2849d366314c5c6) perf: improve serialization performance (#243)
 * [76cac48](https://github.com/numaproj/numalogic/commit/76cac48363f9371ee506c9aaa39f50a67571c8c1) fix: tensor dimension swap instead of view change (#240)
 * [2dfd84c](https://github.com/numaproj/numalogic/commit/2dfd84c2d020dfc1c07b7dea07987eda270384ef) feat: convolutional vae for multivariate time series (#237)
 * [dc18442](https://github.com/numaproj/numalogic/commit/dc18442caccd85bd6ca639110613bfaf88f0ed34) feat: Multivariate threshold using Mahalanobis distance (#234)
 * [466681b](https://github.com/numaproj/numalogic/commit/466681b37193e99b1abae0b1a15b50416035e73d) feat: add thread safety to local cache (#224)

### Contributors

 * Avik Basu
 * Haripriya
 * Jason Zesheng Chen
 * Kushal Batra
 * shrivardhan

## v0.5.0.post1 (2023-07-10)

 * [adebf98](https://github.com/numaproj/numalogic/commit/adebf98e9143ec569901c64f72a106575898b6ef) feat: add thread safety to local cache (#224)

### Contributors

 * Avik Basu

## v0.5.0 (2023-07-06)

 * [b21e246](https://github.com/numaproj/numalogic/commit/b21e2463fd929bdbb7aec4e50918d0b081e000b3) fix (RedisRegistry): avoid overwriting cache with the same key during load  (#223)
 * [0b0daed](https://github.com/numaproj/numalogic/commit/0b0daed7892cce7f08594a2a615acfdf66c04dd0) feat!: add dynamodb registry (#220)
 * [dacae49](https://github.com/numaproj/numalogic/commit/dacae49207ecda788e2615429ae5ecca3811ab99) fix: dataset slicing (#222)
 * [7778ae8](https://github.com/numaproj/numalogic/commit/7778ae856d37a8a6b260532ee7923651df180456) fix: update production key to latest key (#221)
 * [4831180](https://github.com/numaproj/numalogic/commit/483118095fb6e218c861a9b50e4d59b7d5ce11c0) examples: pipeline using block  (#216)
 * [203c100](https://github.com/numaproj/numalogic/commit/203c10040439c9e85a15dce8929293e51179bb23) Upgraded python support for 3.11 (#211)
 * [9ebc32f](https://github.com/numaproj/numalogic/commit/9ebc32f0d22975a897f16ddade40568610c9d266) feat!: introduce numalogic blocks (#206)
 * [4058811](https://github.com/numaproj/numalogic/commit/4058811b3ddc405f7abf45126b21459c6e04d748) fix: redis logging (#209)
 * [b93bce5](https://github.com/numaproj/numalogic/commit/b93bce59d54af1ef57c7fc00f0c1425dcf4ec910) fix: removing logger level setting (#205)
 * [6727296](https://github.com/numaproj/numalogic/commit/67272967ef406114b27fa393c5551c74ebcdd0da) fix: add caching logs (#203)
 * [4411aa7](https://github.com/numaproj/numalogic/commit/4411aa7f0f762a450a925c9546eb66be7de02a55) examples: update with new numalogic and pynumaflow (#202)
 * [2298be4](https://github.com/numaproj/numalogic/commit/2298be4bbac8f120eed654a45ee2656de402edb1) chore!: refactor preproc and postproc into transforms module (#201)
 * [9e838ef](https://github.com/numaproj/numalogic/commit/9e838ef3472544dca3f0bc2e2758c25a76bd520f) fix: Sparse AE for vanilla and conv (#199)
 * [3193159](https://github.com/numaproj/numalogic/commit/3193159eb1f81da6cc3445fe440cb4ef8962e457) fix: registry test_case (#197)
 * [efa1df7](https://github.com/numaproj/numalogic/commit/efa1df75fcf99874ad34ab22c31b6d2a1c876f59) chore!: Auto detect instance type while mlflow model save (#190)
 * [4a1effd](https://github.com/numaproj/numalogic/commit/4a1effd9178f6d5103bef2ea5e5c9f7c432d8d8c) feat: add redis caching (#179)
 * [0f7e6e0](https://github.com/numaproj/numalogic/commit/0f7e6e067a02a0c39954b649085adb027b5b16de) fix: allow import from Registryconfig with optional dependencies (#180)
 * [6c21e95](https://github.com/numaproj/numalogic/commit/6c21e952d37a8fcf7102f095b85fc193255c04d3) fix: stale check; conf lazy imports (#178)
 * [f1909a8](https://github.com/numaproj/numalogic/commit/f1909a8fb2e74d2b6ed563b6e12b7d42cadcb5a5) feat: redis registry (#170)
 * [794ddc6](https://github.com/numaproj/numalogic/commit/794ddc622589a2cf4ef1edb098e51009e23240ea) feat: local memory artifact cache (#165)
 * [03514d6](https://github.com/numaproj/numalogic/commit/03514d6a33a8b34c8a45f19de170167e31e22678) chore!: drop support for python 3.8 (#164)
 * [73bbad2](https://github.com/numaproj/numalogic/commit/73bbad2244fab04f208ff139407c22b766904ecd) feat: first benchmarking using KPI anomaly data (#163)
 * [ed40681](https://github.com/numaproj/numalogic/commit/ed406810cd4de07dc105c3f59313471cefb2058f) feat: support weight decay in optimizers (#161)
 * [cae88b3](https://github.com/numaproj/numalogic/commit/cae88b32af93fc5ed526e8058a4f73684f1dcfbf) chore!: use torch and lightning 2.0 (#159)

### Contributors

 * Avik Basu
 * Kushal Batra
 * Miroslav Boussarov
 * Tarun Chawla

## v0.4.1 (2023-06-20)

 * [79987db](https://github.com/numaproj/numalogic/commit/79987db16ce50bc0d110445b67c059f848f99da7) release: v0.4.1 (#218)

### Contributors

 * Avik Basu

## v0.4.0.post1 (2023-06-06)

 * [b134fa2](https://github.com/numaproj/numalogic/commit/b134fa27a05667c4a51fc5585d0b26618f0c0139) fix: redis logging (#209)

### Contributors

 * Kushal Batra

## v0.4.0 (2023-06-06)

 * [bd050c9](https://github.com/numaproj/numalogic/commit/bd050c9b422cc8cc17d004d03c43b21096a21d47) fix: removing logger level setting (#205)

### Contributors

 * Kushal Batra

## v0.4a1 (2023-06-02)

 * [1f5f458](https://github.com/numaproj/numalogic/commit/1f5f458dab3baf34cb43de7b95e10dc00cc88d71) fix: add caching logs (#203)
 * [ce93191](https://github.com/numaproj/numalogic/commit/ce93191dcc52964c46d1fd78ef9d8c56a646ca77) examples: update with new numalogic and pynumaflow (#202)
 * [fd169cf](https://github.com/numaproj/numalogic/commit/fd169cf2f373b99ef4e5caf78dee49d61f9ede88) chore!: refactor preproc and postproc into transforms module (#201)
 * [a2b00c1](https://github.com/numaproj/numalogic/commit/a2b00c179f46aa2c5c4c17e4e10bb4f0958fbfe7) fix: Sparse AE for vanilla and conv (#199)
 * [5e69f5f](https://github.com/numaproj/numalogic/commit/5e69f5f009ff893273378108d69f9b76917d6a91) fix: registry test_case (#197)

### Contributors

 * Avik Basu
 * Kushal Batra

## v0.4a0 (2023-05-11)

 * [75aea49](https://github.com/numaproj/numalogic/commit/75aea49dfa6d856c7430be00709fd739e734c250) chore!: Auto detect instance type while mlflow model save (#190)
 * [e884b90](https://github.com/numaproj/numalogic/commit/e884b90bf39e72ceb8b95acba59316bbf342e173) feat: add redis caching (#179)
 * [7834730](https://github.com/numaproj/numalogic/commit/7834730465e12cfa12f073209551cedff3af76c6) fix: allow import from Registryconfig with optional dependencies (#180)
 * [25a16f2](https://github.com/numaproj/numalogic/commit/25a16f200733f19a5fc377ed63eee390d261757d) fix: stale check; conf lazy imports (#178)

### Contributors

 * Avik Basu
 * Kushal Batra

## v0.4.dev5 (2023-05-09)

 * [54cffe3](https://github.com/numaproj/numalogic/commit/54cffe374b3e7fd62f91fa9d1b82a8b677837061) fix: optional dependency imports

### Contributors

 * Avik Basu

## v0.4.dev4 (2023-05-09)

 * [8086db1](https://github.com/numaproj/numalogic/commit/8086db121d437ce2bcec0ec11cc338847f197909) fix: stale check; conf lazy imports (#178)
 * [b664e49](https://github.com/numaproj/numalogic/commit/b664e49b5eaa90b43f64a4e7442b5179c7d32149) Prerelease 0.4 (#173)
 * [85fb527](https://github.com/numaproj/numalogic/commit/85fb527be68c325f8308f4d3070fe42b131962ff) chore!: use torch and lightning 2.0

### Contributors

 * Avik Basu

## v0.3.8 (2023-04-18)

 * [3160c2b](https://github.com/numaproj/numalogic/commit/3160c2b4a248f974bc6c6e4893e7c68c1fdd7890) feat: exponential moving average postprocessing (#156)
 * [9de8e4c](https://github.com/numaproj/numalogic/commit/9de8e4cfa7438047b8e7bd22c84bdcf859edc292) fix: validation loss not being logged (#155)

### Contributors

 * Avik Basu

## v0.3.7 (2023-03-27)

 * [b61ac1f](https://github.com/numaproj/numalogic/commit/b61ac1fbd639f482b7ecb661da47254695139299) fix: Tanhscaler nan output for constant feature (#153)
 * [69006eb](https://github.com/numaproj/numalogic/commit/69006ebfb396262d9e31cbc3ada2da7ed274e6a4) Update CODEOWNERS (#151)
 * [6b38465](https://github.com/numaproj/numalogic/commit/6b38465a0ce703215abcb7874211f0def10040f4) feat: more generic convolutional ae (#149)

### Contributors

 * Avik Basu
 * Vigith Maurice

## v0.3.6 (2023-03-22)

 * [cb5509a](https://github.com/numaproj/numalogic/commit/cb5509a2320026ec5bb8e9bca416859b48ab5ea2) add: anomaly sign and return labels for anomalies generated (#146)
 * [b6f63ef](https://github.com/numaproj/numalogic/commit/b6f63efc11fce355367834e707fb255549c06d39) fix: latest model calling (#145)
 * [6ee3446](https://github.com/numaproj/numalogic/commit/6ee34469e73ddd96a96e0a1c13e7fc3d48cea6d8) fix: transition (#144)

### Contributors

 * Kushal Batra

## v0.3.5 (2023-03-09)

 * [ea01b44](https://github.com/numaproj/numalogic/commit/ea01b44a3b6261d4b5ff02e4fd8e5cde53e838ba) feat: Sigmoid threshold (#141)

### Contributors

 * Avik Basu

## v0.3.4 (2023-03-03)

 * [ec8401d](https://github.com/numaproj/numalogic/commit/ec8401d3bbcb6a435af58101eccbfc489c37a4f2) feat: tanh preprocessing (#139)

### Contributors

 * Avik Basu

## v0.3.3 (2023-02-08)

 * [4eae629](https://github.com/numaproj/numalogic/commit/4eae6290bf297dcad6c58b265ef75b4112ff8ffd) fix!: consistency with threshold methods (#138)
 * [2ac1c2f](https://github.com/numaproj/numalogic/commit/2ac1c2f0564d170a52c4259aecb985c4412884d3) feat: static threshold estimator (#136)
 * [d3488c9](https://github.com/numaproj/numalogic/commit/d3488c9e0c6fa017ce965ddb78d41e3ea1209fa8) feat: initial config schema (#135)

### Contributors

 * Avik Basu

## v0.3.2 (2023-01-20)


### Contributors


## v0.3.1 (2023-01-12)

 * [2923718](https://github.com/numaproj/numalogic/commit/2923718b4bfd66ddaf74a84c2dc9b2a48a3a4a1f) fix: unbatch error on certain cases (#131)
 * [232302e](https://github.com/numaproj/numalogic/commit/232302ed134c8447c669b5f56e2684e426cb7e95) fix: pin protobuf to v3.20 for pytorch-lightning (#130)
 * [dca9a7a](https://github.com/numaproj/numalogic/commit/dca9a7a7b39ea229c5f306770967d10aa0285e35) feat!: merge to release v0.3 (#119)
 * [df20591](https://github.com/numaproj/numalogic/commit/df20591f5a3b45117525c4331e5668c4289bd118) fix: sklearn base import for scikit learn v1.2 (#112)
 * [d2c6293](https://github.com/numaproj/numalogic/commit/d2c6293f594ed67cc8a6ddabce1f8f0bdd84249f) fix: change example pipeline mlflow port (#96)
 * [f9c74c9](https://github.com/numaproj/numalogic/commit/f9c74c90c37fca24db3465c24326a639e93bc802) fix: allow only patch updates in torch version due to cuda build errors on mac (#90)
 * [0a5fcdf](https://github.com/numaproj/numalogic/commit/0a5fcdf6c1ea4732b2c6c97f57d983751cfe2956) fix_readme: mention namespace name when applying the pipeline (#88)

### Contributors

 * Avik Basu
 * Kushal Batra

## v0.3.0a1 (2022-12-22)

 * [88d26ec](https://github.com/numaproj/numalogic/commit/88d26ec000ee719f967dc90f4f78f57784b1db7e) feat!: convert AE variants to lightning modules (#110)
 * [ed94615](https://github.com/numaproj/numalogic/commit/ed9461517f68192d275d81ef38727bcbe62e887f) fix: fix and clean mlflow test cases (#109)

### Contributors

 * Avik Basu
 * Kushal Batra

## v0.3.0a0 (2022-12-08)

 * [78cf5b4](https://github.com/numaproj/numalogic/commit/78cf5b4e26ffd6dffa5e0dc364cbd51b220bd928) fix: fix pipeline for 0.3 release (#106)
 * [709553f](https://github.com/numaproj/numalogic/commit/709553f4ae879f7e786634b02867966756706174) fix: fix mlflow test cases (#98)
 * [701812e](https://github.com/numaproj/numalogic/commit/701812efed6742915d9cf57aef985127b3cc449d) feat!: disentangle threshold selection from the main model  (#89)

### Contributors

 * Avik Basu
 * Kushal Batra

## v0.3.0 (2023-01-05)

 * [dca9a7a](https://github.com/numaproj/numalogic/commit/dca9a7a7b39ea229c5f306770967d10aa0285e35) feat!: merge to release v0.3 (#119)

### Contributors

 * Avik Basu

## v0.2.10 (2023-01-06)

 * [c00bb14](https://github.com/numaproj/numalogic/commit/c00bb140dc481d90b70f98e8fbb1968db151b9ca) fix: Upgrade torch to 1.13.1 (#128)

### Contributors

 * Avik Basu

## v0.2.9 (2022-12-21)

 * [df20591](https://github.com/numaproj/numalogic/commit/df20591f5a3b45117525c4331e5668c4289bd118) fix: sklearn base import for scikit learn v1.2 (#112)

### Contributors

 * Avik Basu

## v0.2.8 (2022-11-29)

 * [7d5075d](https://github.com/numaproj/numalogic/commit/7d5075ddda5950fef772cf07ffdb8f949bed589f) remove mlflow full
 * [05d6071](https://github.com/numaproj/numalogic/commit/05d6071a1de4ca85ed8095bab7654c2e77f366f5) fix: have mlflow-server as an optional extra
 * [3c5bc83](https://github.com/numaproj/numalogic/commit/3c5bc8371a50a98a7ffc634b034692cc53ba597c) fix: lock file
 * [d2c6293](https://github.com/numaproj/numalogic/commit/d2c6293f594ed67cc8a6ddabce1f8f0bdd84249f) fix: change example pipeline mlflow port (#96)

### Contributors

 * Avik Basu
 * Kushal Batra

## v0.2.7 (2022-11-14)

 * [f9c74c9](https://github.com/numaproj/numalogic/commit/f9c74c90c37fca24db3465c24326a639e93bc802) fix: allow only patch updates in torch version due to cuda build errors on mac (#90)
 * [0a5fcdf](https://github.com/numaproj/numalogic/commit/0a5fcdf6c1ea4732b2c6c97f57d983751cfe2956) fix_readme: mention namespace name when applying the pipeline (#88)
 * [537fae5](https://github.com/numaproj/numalogic/commit/537fae56577954391d10b134cb318d388c1a7dd7) fix: AutoencoderPipeline logged loss mean (#55)
 * [22f8e5d](https://github.com/numaproj/numalogic/commit/22f8e5d79183d848f68c7cbeb2d88126fb326d03) chore!: make mlflow as an optional dependency (#47)

### Contributors

 * Avik Basu
 * Kushal Batra
 * diego-ponce

## v0.2.6 (2022-10-17)

 * [5703d1b](https://github.com/numaproj/numalogic/commit/5703d1b75d642242dc5c521a2e85e9cbb2527923) fix: update readme with optional mlflow dependency
 * [702f3b4](https://github.com/numaproj/numalogic/commit/702f3b45bea0890de2db6a2b61a44dd0a675d3be) fix: install extras in workflows
 * [8969e80](https://github.com/numaproj/numalogic/commit/8969e80e1831e0b6823a650f400d954de2e76c22) chore!: make mlflow as an optional dependency
 * [9cc97cb](https://github.com/numaproj/numalogic/commit/9cc97cbb7161fa61108288c0381a10b802114b4d) feat: resume training parameter (#40)

### Contributors

 * Avik Basu
 * Kushal Batra

## v0.2.5 (2022-09-28)


### Contributors


## v0.2.4 (2022-09-21)

 * [f0b0d3c](https://github.com/numaproj/numalogic/commit/f0b0d3c98bb12722dd9b3c5d064e3d0330008d07) fix: loading secondary artifacts (#16)
 * [0c506bd](https://github.com/numaproj/numalogic/commit/0c506bd53e00f6d0c9abfc722b9b3bb57c9cd59f) chore (#19)
 * [c845d09](https://github.com/numaproj/numalogic/commit/c845d099ca038b761e78f306b4e33f4924737197) Update README.md
 * [e8be7c5](https://github.com/numaproj/numalogic/commit/e8be7c513db2c1f6db1b074fd3832357fa113d1a) fix: pypi auto publish workflow (#18)
 * [fa22031](https://github.com/numaproj/numalogic/commit/fa220317229594bce67b37b13e94a22001b1a8e2) [Chore] Update README (#17)

### Contributors

 * Avik Basu
 * Kushal Batra
 * Saradhi Sreegiriraju
 * Vigith Maurice
 * amitkalamkar

## v0.2.3 (2022-08-16)

 * [6847211](https://github.com/numaproj/numalogic/commit/68472118ab3e64ca4f10ddc8f255e41dfcef3036) workflows: add pypi publish and auto release generation (#14)
 * [040584f](https://github.com/numaproj/numalogic/commit/040584f8bffb395504a3f0be95e5a74e5f4915e4) feat: adding feature for retaining fixed number of stale model (#13)

### Contributors

 * Avik Basu
 * Kushal Batra

## v0.2.2 (2022-08-03)

 * [a5ef072](https://github.com/numaproj/numalogic/commit/a5ef072ad6742dfcb025b43b90803c26d4915ff8) feat: Add support for storing preproc artifacts (scondary artifact) i… (#11)

### Contributors

 * Kushal Batra

## v0.2.1 (2022-07-21)


### Contributors


## v0.2.0 (2022-07-21)

 * [5314070](https://github.com/numaproj/numalogic/commit/5314070b927f5334f8022b6c5fed843dd248a1d4) feat: add transformers model (#8)

### Contributors

 * Kushal Batra

