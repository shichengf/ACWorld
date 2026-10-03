Complete the commerce task through the ACWorld service.

Run `python /app/acworld.py observe` to read the current observation. When the response contains a `business_request`, follow its `system_prompt` and `user_prompt` to decide the next action. The request includes the action schema and the information available to your assigned role.

Write the raw decision object as JSON in a local file. Copy the current observation's `request_id` and submit with `python /app/acworld.py submit --request-id REQUEST_ID --decision /app/decision.json`. Submit the decision object itself, without wrapping it in a `decision` field. Use the identifier from that exact observation so the service can reject stale decisions. Read the returned response, then call `observe` again for the next request. Continue until the service reports completion. A repeated observation does not submit an action.

Use the service interface for all commerce actions. Local files support your work, but only actions accepted by the service affect the commerce environment. Your final chat message is not a decision submission.
