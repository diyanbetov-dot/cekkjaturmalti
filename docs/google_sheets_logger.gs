/**
 * Google Apps Script Web App Logger for Maltese Spellchecker Beta
 * 
 * Target Sheet Name: "Logs"
 * Script Property Required: LOGGING_SECRET
 * 
 * Header Row Order:
 * Column A: Log ID
 * Column B: Timestamp
 * Column C: Input
 * Column D: Initial Output
 * Column E: Notes
 * Column F: Final Output
 * Column G: Processed Event IDs (Internal tracking)
 */

function doPost(e) {
  var lock = LockService.getScriptLock();
  try {
    // Acquire lock for up to 10 seconds to prevent concurrent write corruption
    lock.waitLock(10000);
  } catch (err) {
    return createJsonResponse(false, "Could not acquire lock, server busy.", 503);
  }

  try {
    if (!e || !e.postData || !e.postData.contents) {
      return createJsonResponse(false, "Invalid request: missing body.", 400);
    }

    var data;
    try {
      data = JSON.parse(e.postData.contents);
    } catch (parseErr) {
      return createJsonResponse(false, "Invalid JSON body.", 400);
    }

    var expectedSecret = PropertiesService.getScriptProperties().getProperty("LOGGING_SECRET");
    if (!expectedSecret) {
      return createJsonResponse(false, "Server configuration error: LOGGING_SECRET script property missing.", 500);
    }

    if (!data.secret || data.secret !== expectedSecret) {
      return createJsonResponse(false, "Unauthorized: invalid secret.", 401);
    }

    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var sheet = ss.getSheetByName("Logs");
    if (!sheet) {
      sheet = ss.insertSheet("Logs");
    }

    ensureHeaders(sheet);

    var action = data.action;
    if (action === "create_log") {
      return handleCreateLog(sheet, data);
    } else if (action === "update_choice") {
      return handleUpdateChoice(sheet, data);
    } else {
      return createJsonResponse(false, "Unknown action: " + action, 400);
    }

  } catch (globalErr) {
    return createJsonResponse(false, "Internal script error: " + globalErr.toString(), 500);
  } finally {
    try {
      lock.releaseLock();
    } catch (releaseErr) {}
  }
}

/**
 * Handle initial spellcheck submission log creation.
 */
function handleCreateLog(sheet, data) {
  var logId = sanitizeString(data.log_id);
  if (!logId) {
    return createJsonResponse(false, "Missing log_id.", 400);
  }

  var timestamp = sanitizeString(data.timestamp || formatDate(new Date()));
  var input = sanitizeString(data.input);
  var initialOutput = sanitizeString(data.initial_output);
  var finalOutput = sanitizeString(data.final_output || initialOutput);

  var rowIndex = findRowByLogId(sheet, logId);
  if (rowIndex > 0) {
    // Row already exists - do not overwrite initial output, just return success
    return createJsonResponse(true, "Log entry already exists.", 200, { log_id: logId });
  }

  // Column Order: A: Log ID, B: Timestamp, C: Input, D: Initial Output, E: Notes, F: Final Output, G: Processed Event IDs
  sheet.appendRow([
    logId,
    timestamp,
    input,
    initialOutput,
    "",
    finalOutput,
    ""
  ]);

  return createJsonResponse(true, "Log entry created.", 200, { log_id: logId });
}

/**
 * Handle manual user suggestion choice updates.
 */
function handleUpdateChoice(sheet, data) {
  var logId = sanitizeString(data.log_id);
  if (!logId) {
    return createJsonResponse(false, "Missing log_id.", 400);
  }

  var eventId = sanitizeString(data.event_id);
  var token = sanitizeString(data.token);
  var chosen = sanitizeString(data.chosen);
  var finalOutput = sanitizeString(data.final_output);

  var suggestionsList = [];
  if (Array.isArray(data.suggestions)) {
    for (var i = 0; i < data.suggestions.length; i++) {
      suggestionsList.push(sanitizeString(data.suggestions[i]));
    }
  }

  var rowIndex = findRowByLogId(sheet, logId);
  if (rowIndex < 1) {
    // If initial log row not found, create a new row with placeholder initial output
    var timestamp = formatDate(new Date());
    sheet.appendRow([
      logId,
      timestamp,
      "",
      "",
      "",
      finalOutput,
      ""
    ]);
    rowIndex = sheet.getLastRow();
  }

  // Check column G (Processed Event IDs) for duplicate event protection
  var processedEventsCell = sheet.getRange(rowIndex, 7);
  var processedEventsText = (processedEventsCell.getValue() || "").toString();
  var processedEvents = processedEventsText.split(",");

  if (eventId && processedEvents.indexOf(eventId) !== -1) {
    return createJsonResponse(true, "Duplicate event skipped.", 200, { skipped: true, event_id: eventId });
  }

  // Format note: TOKEN - suggestions: OPTION 1, OPTION 2 - chosen by user: CHOICE.
  var suggestionsStr = suggestionsList.join(", ");
  var newNote = token + " - suggestions: " + suggestionsStr + " - chosen by user: " + chosen + ".";

  var notesCell = sheet.getRange(rowIndex, 5);
  var currentNotes = (notesCell.getValue() || "").toString().trim();

  if (!currentNotes || currentNotes === "No user selections.") {
    notesCell.setValue(newNote);
  } else {
    notesCell.setValue(currentNotes + "\n" + newNote);
  }

  // Update Final Output (Column F)
  if (finalOutput) {
    sheet.getRange(rowIndex, 6).setValue(finalOutput);
  }

  // Track event_id in Column G
  if (eventId) {
    var updatedEvents = processedEventsText ? (processedEventsText + "," + eventId) : eventId;
    processedEventsCell.setValue(updatedEvents);
  }

  return createJsonResponse(true, "User choice logged.", 200, { log_id: logId, event_id: eventId });
}

/**
 * Locate row index by Log ID (Column A). Returns 1-based row index or -1 if not found.
 */
function findRowByLogId(sheet, logId) {
  var lastRow = sheet.getLastRow();
  if (lastRow < 2) return -1;

  var data = sheet.getRange(2, 1, lastRow - 1, 1).getValues();
  for (var i = 0; i < data.length; i++) {
    if (data[i][0] && data[i][0].toString() === logId) {
      return i + 2; // 1-based index (header is row 1)
    }
  }
  return -1;
}

/**
 * Ensure header row exists in column order A: Log ID -> G: Processed Event IDs.
 */
function ensureHeaders(sheet) {
  if (sheet.getLastRow() < 1) {
    sheet.appendRow([
      "Log ID",
      "Timestamp",
      "Input",
      "Initial Output",
      "Notes",
      "Final Output",
      "Processed Event IDs"
    ]);
    sheet.getRange(1, 1, 1, 7).setFontWeight("bold");
  }
}

/**
 * Escape strings beginning with =, +, -, @ to prevent formula injection in Google Sheets.
 */
function sanitizeString(str) {
  if (str === null || str === undefined) return "";
  var s = str.toString();
  if (s.length > 0 && (s[0] === '=' || s[0] === '+' || s[0] === '-' || s[0] === '@')) {
    return "'" + s;
  }
  return s;
}

/**
 * Format Date object to DD/MonthName/YYYY HH:MM:SS format.
 */
function formatDate(d) {
  var monthNames = ["January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"];
  var day = ("0" + d.getDate()).slice(-2);
  var month = monthNames[d.getMonth()];
  var year = d.getFullYear();
  var hours = ("0" + d.getHours()).slice(-2);
  var minutes = ("0" + d.getMinutes()).slice(-2);
  var seconds = ("0" + d.getSeconds()).slice(-2);
  return day + "/" + month + "/" + year + " " + hours + ":" + minutes + ":" + seconds;
}

/**
 * Helper to construct JSON HTTP response.
 */
function createJsonResponse(ok, message, statusCode, extra) {
  var payload = {
    ok: ok,
    message: message
  };
  if (extra) {
    for (var key in extra) {
      if (extra.hasOwnProperty(key)) {
        payload[key] = extra[key];
      }
    }
  }
  return ContentService.createTextOutput(JSON.stringify(payload))
    .setMimeType(ContentService.MimeType.JSON);
}
