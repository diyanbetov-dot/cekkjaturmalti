/*
 * Google Apps Script extension for Cekkjatur tal-Malti feedback reports.
 *
 * Add this file to the Apps Script project already receiving create_log and
 * update_choice. After the existing secret check in doPost(e), add:
 *
 *   if (payload.action === "feedback_report") {
 *     return handleFeedbackReport_(payload);
 *   }
 *
 * Run setupFeedbackReporting() once from the Apps Script editor, approve the
 * requested Sheets, Drive, and Mail permissions, then redeploy the web app.
 */

const FEEDBACK_CONFIG = Object.freeze({
  // Set this to the exact name of the sheet containing the spellcheck logs.
  logSheetName: "Spellcheck Logs",
  recipientEmail: "cekkjaturemalti@gmail.com",

  // Create a private Drive folder for screenshots and paste its folder ID here.
  driveFolderId: "REPLACE_WITH_DRIVE_FOLDER_ID",

  // Existing log table layout: log ID in column A and log fields in A:H.
  logIdColumn: 1,
  existingLogColumnCount: 8,
  feedbackFirstColumn: 9,
  maximumVisibleRowHeight: 120
});

const FEEDBACK_HEADERS = Object.freeze([
  "Feedback timestamp",
  "Reporter email",
  "Feedback subject",
  "Feedback message",
  "Screenshot",
  "Reported word",
  "Page URL"
]);

function setupFeedbackReporting() {
  const sheet = getFeedbackLogSheet_();
  sheet
    .getRange(1, FEEDBACK_CONFIG.feedbackFirstColumn, 1, FEEDBACK_HEADERS.length)
    .setValues([FEEDBACK_HEADERS])
    .setFontWeight("bold")
    .setWrapStrategy(SpreadsheetApp.WrapStrategy.WRAP);

  // Keep paragraphs inside A:H so they cannot visually cover I onward.
  sheet.setColumnWidths(1, FEEDBACK_CONFIG.existingLogColumnCount, 220);
  sheet.setColumnWidth(1, 165);
  sheet.setColumnWidth(2, 165);
  sheet.setColumnWidths(
    FEEDBACK_CONFIG.feedbackFirstColumn,
    FEEDBACK_HEADERS.length,
    180
  );
  sheet.setColumnWidth(FEEDBACK_CONFIG.feedbackFirstColumn + 3, 320);
  sheet.setColumnWidth(FEEDBACK_CONFIG.feedbackFirstColumn + 4, 240);

  // Pre-format the available grid. Future values inherit wrapping and cannot
  // spill visually into columns I onward.
  const availableRows = Math.max(1, sheet.getMaxRows() - 1);
  sheet
    .getRange(
      2,
      1,
      availableRows,
      FEEDBACK_CONFIG.existingLogColumnCount + FEEDBACK_HEADERS.length
    )
    .setWrapStrategy(SpreadsheetApp.WrapStrategy.WRAP)
    .setVerticalAlignment("top");
  if (sheet.getLastRow() > 1) {
    formatStoredRows_(sheet, 2, sheet.getLastRow() - 1);
  }
}

function handleFeedbackReport_(payload) {
  try {
    const email = requireFeedbackText_(payload.email, "email", 254);
    const subject = requireFeedbackText_(payload.subject, "subject", 200)
      .replace(/[\r\n]+/g, " ");
    const message = requireFeedbackText_(payload.message, "message", 10000);
    const timestamp = String(payload.timestamp || new Date().toISOString());
    const logId = String(payload.log_id || "").slice(0, 100);
    const reportedWord = String(payload.reported_word || "").slice(0, 200);
    const pageUrl = String(payload.page_url || "").slice(0, 1000);

    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) {
      throw new Error("Invalid reporter email address.");
    }

    const sheet = getFeedbackLogSheet_();
    const attachment = decodeFeedbackScreenshot_(payload);
    let screenshotUrl = "";
    if (attachment) {
      const folder = DriveApp.getFolderById(FEEDBACK_CONFIG.driveFolderId);
      const file = folder.createFile(attachment);
      screenshotUrl = file.getUrl();
    }

    let row = findLogRow_(sheet, logId);
    if (!row) {
      row = Math.max(2, sheet.getLastRow() + 1);
      if (logId) {
        sheet.getRange(row, FEEDBACK_CONFIG.logIdColumn).setValue(logId);
      }
    }

    sheet
      .getRange(
        row,
        FEEDBACK_CONFIG.feedbackFirstColumn,
        1,
        FEEDBACK_HEADERS.length
      )
      .setValues([[
        timestamp,
        email,
        subject,
        message,
        screenshotUrl,
        reportedWord,
        pageUrl
      ]]);
    formatStoredRows_(sheet, row, 1);

    const emailBody = [
      message,
      "",
      "Reporter: " + email,
      reportedWord ? "Reported word: " + reportedWord : "",
      logId ? "Spellcheck log ID: " + logId : "",
      pageUrl ? "Page: " + pageUrl : "",
      screenshotUrl ? "Drive screenshot: " + screenshotUrl : ""
    ].filter(String).join("\n");

    const mailOptions = {
      to: FEEDBACK_CONFIG.recipientEmail,
      replyTo: email,
      subject: "[Cekkjatur] " + subject,
      body: emailBody,
      name: "Cekkjatur tal-Malti"
    };
    if (attachment) {
      mailOptions.attachments = [attachment];
    }
    MailApp.sendEmail(mailOptions);

    return feedbackJsonResponse_({
      ok: true,
      row: row,
      screenshot_url: screenshotUrl
    });
  } catch (error) {
    console.error("Feedback report failed", error);
    return feedbackJsonResponse_({
      ok: false,
      error: String(error && error.message ? error.message : error)
    });
  }
}

function getFeedbackLogSheet_() {
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = spreadsheet.getSheetByName(FEEDBACK_CONFIG.logSheetName);
  if (!sheet) {
    throw new Error(
      'Sheet "' + FEEDBACK_CONFIG.logSheetName + '" was not found.'
    );
  }
  return sheet;
}

function findLogRow_(sheet, logId) {
  if (!logId || sheet.getLastRow() < 2) {
    return 0;
  }
  const match = sheet
    .getRange(
      2,
      FEEDBACK_CONFIG.logIdColumn,
      sheet.getLastRow() - 1,
      1
    )
    .createTextFinder(logId)
    .matchEntireCell(true)
    .findNext();
  return match ? match.getRow() : 0;
}

function formatStoredRows_(sheet, firstRow, rowCount) {
  if (rowCount <= 0) {
    return;
  }
  sheet
    .getRange(
      firstRow,
      1,
      rowCount,
      FEEDBACK_CONFIG.existingLogColumnCount
    )
    .setWrapStrategy(SpreadsheetApp.WrapStrategy.WRAP)
    .setVerticalAlignment("top");
  sheet
    .getRange(
      firstRow,
      FEEDBACK_CONFIG.feedbackFirstColumn,
      rowCount,
      FEEDBACK_HEADERS.length
    )
    .setWrapStrategy(SpreadsheetApp.WrapStrategy.WRAP)
    .setVerticalAlignment("top");

  sheet.autoResizeRows(firstRow, rowCount);
  for (let row = firstRow; row < firstRow + rowCount; row += 1) {
    if (sheet.getRowHeight(row) > FEEDBACK_CONFIG.maximumVisibleRowHeight) {
      sheet.setRowHeight(row, FEEDBACK_CONFIG.maximumVisibleRowHeight);
    }
  }
}

function decodeFeedbackScreenshot_(payload) {
  const dataUrl = String(payload.screenshot_data_url || "");
  if (!dataUrl) {
    return null;
  }
  const match = dataUrl.match(
    /^data:(image\/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=\r\n]+)$/
  );
  if (!match) {
    throw new Error("Invalid screenshot data.");
  }
  const bytes = Utilities.base64Decode(match[2]);
  if (bytes.length > 2000000) {
    throw new Error("Screenshot exceeds the 2 MB limit.");
  }
  const filename = String(
    payload.screenshot_filename || "cekkjatur-report.jpg"
  ).replace(/[^A-Za-z0-9._-]/g, "_");
  return Utilities.newBlob(bytes, match[1], filename);
}

function requireFeedbackText_(value, fieldName, maximumLength) {
  const text = String(value || "").trim();
  if (!text || text.length > maximumLength) {
    throw new Error(
      fieldName + " must contain 1 to " + maximumLength + " characters."
    );
  }
  return text;
}

function feedbackJsonResponse_(payload) {
  return ContentService
    .createTextOutput(JSON.stringify(payload))
    .setMimeType(ContentService.MimeType.JSON);
}
