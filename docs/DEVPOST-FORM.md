# Devpost form for Raised: paste field by field

Project name: Raised

Elevator pitch (tagline, 200 characters max):
Raised: Transforms textbook charts into accessible data: a sound table and tactile braille graphic, verifying each number’s accuracy.

Track: Best Apps and Agents

Built with (type each tag): Nebius Token Factory, NVIDIA Nemotron 3.5 Super VL, NVIDIA Nemotron 3 Ultra, NVIDIA Nemotron 3.5 Lightning, Python, FastAPI, Tesseract, liblouis, Modal

Try it out links:
- https://m-simakhov--raised-web.us-east.modal.run
- https://github.com/msimakhov-star/raised

Video demo link: https://youtu.be/7ReJapiM6pU   (publish it as Public first)

City (Builders & Brews): leave blank unless Mike attended a meetup in person.

Did the project exist before 26 Aug? No.

--- About the project (paste into the main story box) ---

## Inspiration

This project aims to create accessible representations of data charts found within textbook PDF documents. Our goal is to provide alternative formats suitable for visually impaired individuals.

## What it does

Raised takes a textbook PDF or chart image as input. It identifies each data chart within the document. The system then generates several outputs: a checked data table, a detailed textual description of the chart, alt text for screen reader accessibility, a sound explorer controlled by arrow keys, a tactile graphic conforming to the BANA Guidelines and Standards for Tactile Graphics (2010), and a braille text file.

## How we built it

The system utilises a multi-stage process. Initially, NVIDIA Nemotron 3.5 Super VL is used to interpret the meaning of each chart and extract a first value for each data point. Simultaneously, plain code reads the chart’s tick labels using Optical Character Recognition (OCR), adjusts the axis scale, and measures lines, bars, and points using colour information. When the readings from the two systems disagree, both are redrawn over the original image, and the result that matches the pixel data is retained. Points where the readings differ are flagged for review by a transcriber.

NVIDIA Nemotron 3 Ultra then writes the description of the chart and verifies the book’s caption against the generated data. The code rejects any generated numerical values not found within the data table. NVIDIA Nemotron 3.5 Lightning performs sentence checking, shortens labels for braille transcription, and generates alt text. All model calls are executed on the Nebius Token Factory.

The software is built using Python, FastAPI, Tesseract OCR, liblouis, and Modal. The tactile layout design adheres to the BANA Guidelines and Standards for Tactile Graphics (2010).

## Challenges we ran into

Measuring lines that cross or are obscured by legends presented a significant challenge. Charts with log axes containing few labels also proved difficult to process accurately. Fitting braille labels without causing overlaps required careful design and algorithmic adjustments.

## Accomplishments that we’re proud of

Across 350 test charts, we achieved no label overlaps in the tactile layouts. Every generated output is labelled as a draft, intended for review and correction by a certified transcriber.

## What we learned

We found that a vision model is most effective when used for interpreting the meaning of a chart. Conversely, precise measurement is best achieved through code that allows for thorough verification.

## What’s next

Our next steps involve testing the system with teachers of visually impaired students and certified transcribers. We also plan to expand the range of chart types supported. Measurements, using pixel measurement alone, have a median error of 0.24% of the axis range, when measured on 300 synthetic charts with known values.

--- Feedback on Nebius and NVIDIA tools (paste into the feedback field) ---

## Nebius Token Factory

The Nebius Token Factory platform provided access to models for use within the hackathon. The account onboarding process required payment card details before an API key could be generated. The public model catalogue at tokenfactory.nebius.com/api/public/models_info listed precise model identifiers and associated costs. Model identifiers exhibited inconsistent formatting: Nemotron-3.5-Super-VL, Nemotron-3_5-Lightning, and Nemotron-3-Ultra-550b-a55b were observed. The system accepted an OpenAI-style JSON schema wrapper for requests, but not a bare schema. The ‘n’ parameter was correctly implemented, returning three model choices in a single request.

## NVIDIA models

Testing revealed a rate limit of 1000 requests and 2,000,000 tokens per minute, without any 429 errors. The Nemotron 3.5 Super VL model accurately processed ten test charts, demonstrating a median error of 0.8% of the axis range when reading values. A structured request to Nemotron 3 Ultra responded in approximately one second; Nemotron 3.5 Lightning took approximately ten seconds for a similar structured response in a single test.

## What we would change

The documentation for Nemotron 3.5 Super VL lists video as an input type but does not specify the request format. There is no defined spend cap on the account; the application enforces its own daily budget. On a repeated identical image prompt, the usage field reported 0 cached tokens. Charts with axes drawn in the same colour as the data line necessitated additional processing on our side. We would build with Token Factory again.
