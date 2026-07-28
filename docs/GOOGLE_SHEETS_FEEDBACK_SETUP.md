# Google Sheets feedback setup

The website sends feedback to Flask at `/submit-feedback`. Flask then reuses
the existing secret-protected Google Apps Script endpoint. The browser never
receives `SPELLCHECK_LOG_SECRET`.

## Apps Script changes

1. Open the Apps Script project currently receiving `create_log` and
   `update_choice`.
2. Add a new script file and paste the contents of
   `docs/google_apps_script_feedback.gs`.
3. Change `FEEDBACK_CONFIG.logSheetName` to the exact name of the existing log
   sheet.
4. Create a private Google Drive folder for screenshots. Copy the folder ID
   from its URL and replace `REPLACE_WITH_DRIVE_FOLDER_ID`.
5. Confirm that `FEEDBACK_CONFIG.logIdColumn` matches the column containing
   `log_id`. The supplied value `1` means column A.
6. In the existing `doPost(e)`, after parsing the JSON and validating the
   existing secret, add:

   ```javascript
   if (payload.action === "feedback_report") {
     return handleFeedbackReport_(payload);
   }
   ```

7. Immediately after the existing `create_log` action appends a row, add:

   ```javascript
   formatStoredRows_(sheet, sheet.getLastRow(), 1);
   ```

   Replace `sheet` if your existing handler uses a different variable name.
   This auto-sizes short entries and caps long paragraphs at 120 pixels.
8. Run `setupFeedbackReporting()` once from the Apps Script editor. Approve
   access to Sheets, Drive, and Mail when Google asks.
9. Deploy a new version of the existing web app. Keep access compatible with
   the current Flask logger deployment.

The setup reserves columns A:H for current logs and I:O for feedback. Existing
and future paragraph cells wrap at a fixed width and rows are capped at 120
pixels, so their text cannot visually spill across the feedback columns.
Selecting a cell still exposes its complete stored contents.

## Cloud Run settings

These existing environment variables must remain configured:

```text
SPELLCHECK_BETA_LOGGING=true
SPELLCHECK_LOG_URL=https://script.google.com/macros/s/YOUR_DEPLOYMENT_ID/exec
SPELLCHECK_LOG_SECRET=YOUR_EXISTING_SECRET
```

No Drive or email credentials belong in the HTML. Apps Script sends each report
to `cekkjaturemalti@gmail.com`, uses the visitor's address as `Reply-To`, saves
the screenshot privately in Drive, and places its Drive URL in the Sheet.
