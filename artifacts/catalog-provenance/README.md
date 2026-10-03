# ACWorld catalog provenance

Spring Brand supplied the source catalog to the research team. Its platform collected public product information from six retailers on 21 May 2026. Normalization produced 1,082 canonical listings.

The data provider's website is [Spring Brand](https://springbrand.ai/). Website fields in this release identify the data provider.

| File | Contents |
| --- | --- |
| `source_summary.json` | Provider information, collection date, and counts for sources A–F. |
| `source_urls.csv` | The provider website, source labels, and catalog record identifiers for the 1,082 canonical listings. |
| `field_missingness.csv` | Missing field counts in the source catalog, with each row identifying its population. |

Catalog records use source labels A–F and identifiers of the form `catalog-0001`. Versioned task builders define synthetic identifiers, constraints, and transaction conditions for the benchmark, including inventory, delivery, preferences, and transaction histories. The paper's 200 tasks and scoring rules are provided in ACWorld v1.0.0.
