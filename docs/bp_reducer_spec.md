# Design to reduce the boilerplate

Goal: Create a spec for refactory the package in order to Reduce the amount of boilerplate for this package.

After use this package for a while, we've realized that just a few node are often used, which are:

- ModelCall: call a model in order to get a structured data fullfiled back. When it comes to chaining flows, this is the most frequent. The way it is, that consists:
    - LLM call with tool (with tool choice) which is just the datatype (schema definition) and the handler of the tools just return the object after validated.
    - So we don't need to write all that boilerplate, just write it just the function for transform 

- CompletionCall:
    - LLM Call only expect the content property

- ModelTools: llm call to evaluate whether call tools. So the steps are
    - LLM call
    - for each tool -> execute the handler
    - return the list of objects returned to the handlers

- ToolsOnly: this is about the execution just the handler without llm call, returning the object of the handler

In order to do that, you must

- Create section for a full analysis of the state of the project about the "Actors" call, use the index.py file to see an example of a application
- Create another section with the idea of the refactory
- Create another section with design decisions, for each design, define a question, options, for each option pros and cons, a recommended options and a brief rationale then a placeholder for me to decide and comment. 
- If you have overall suggestion for the design or refactory stuff, create another section with the same question stuff of the previous section