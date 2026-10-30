## Nebius Token Factory

The Nebius Token Factory platform provided access to models for use within the hackathon. The account onboarding process required payment card details before an API key could be generated. The public model catalogue at tokenfactory.nebius.com/api/public/models_info listed precise model identifiers and associated costs. Model identifiers exhibited inconsistent formatting: Nemotron-3.5-Super-VL, Nemotron-3_5-Lightning, and Nemotron-3-Ultra-550b-a55b were observed. The system accepted an OpenAI-style JSON schema wrapper for requests, but not a bare schema. The ‘n’ parameter was correctly implemented, returning three model choices in a single request.

## NVIDIA models

Testing revealed a rate limit of 1000 requests and 2,000,000 tokens per minute, without any 429 errors. The Nemotron 3.5 Super VL model accurately processed ten test charts, demonstrating a median error of 0.8% of the axis range when reading values. A structured request to Nemotron 3 Ultra responded in approximately one second; Nemotron 3.5 Lightning took approximately ten seconds for a similar structured response in a single test.

## What we would change

The documentation for Nemotron 3.5 Super VL lists video as an input type but does not specify the request format. There is no defined spend cap on the account; the application enforces its own daily budget. On a repeated identical image prompt, the usage field reported 0 cached tokens. Charts with axes drawn in the same colour as the data line necessitated additional processing on our side. We would build with Token Factory again.
