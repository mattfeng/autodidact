Generate a system prompt for creating a deep learning problem set. The prompt should result in a problem set that:

- Is written in second person and directly addresses the student.
- Breaks down the overall assignment into subproblems, and sub-subproblems as necessary.
- Does not talk about submissions/rubrics, only correctness.
- Provides theory and context, natural language pseudocode with implementation
  details, and simple test cases.
- Provides function and class headers with docstrings, but does not implement any bodies. Exceptions include dataclasses, or helper routines, scripts, and test cases generated for the student.
- Provides a directory structure outline.
- Provides helper scripts and test cases where it's not critical for the student's understanding of the topic. This include dataset downloading and preparation.
- Instructor-provided scripts can and should use external libraries, especially where speed matters.
- Is specific about which files, datasets, commands, etc. to create, edit, and use. The commands must actually work.
- Uses `pytest` for testing.
- Minimizes uses of external libraries unless absolutely necessary or performs functionality that is not critical for the student's understanding.
- For tasks that are not central for the student's learning and understanding,
  provides very specific instructions to avoid ambiguity and wasted time.
- Provides a table of contents at the start of the problem set.
