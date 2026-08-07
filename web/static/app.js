document.getElementById("checkBtn").addEventListener("click", async () => {
    const text = document.getElementById("inputText").value;
    const res = await fetch("/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
    });
    const data = await res.json();
    document.getElementById("outputText").innerText = data.corrected_text;
    document.getElementById("latency").innerText = `Latency: ${data.latency_ms} ms`;
});

document.getElementById("clearBtn").addEventListener("click", () => {
    document.getElementById("inputText").value = "";
    document.getElementById("outputText").innerText = "";
    document.getElementById("latency").innerText = "Latency: 0 ms";
});

document.getElementById("copyBtn").addEventListener("click", () => {
    const text = document.getElementById("outputText").innerText;
    navigator.clipboard.writeText(text);
});
