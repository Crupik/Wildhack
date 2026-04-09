document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predictForm");
  const submitBtn = document.getElementById("submitBtn");
  const loading = document.getElementById("loading");
  const errorBox = document.getElementById("errorBox");

  const resultCard = document.getElementById("resultCard");
  const placeholderCard = document.getElementById("placeholderCard");

  const routeEl = document.getElementById("result_route_id");
  const officeEl = document.getElementById("result_office_from_id");
  const predEl = document.getElementById("result_predicted_target_2h");
  const capacityEl = document.getElementById("result_required_capacity");
  const allocatedEl = document.getElementById("result_allocated_vehicles");
  const statusBadge = document.getElementById("statusBadge");

  const TRUCK_LABELS = {
  "10t": "10 тонн",
  "20t_82": "20 тонн (82 м³)",
  "20t_90": "20 тонн (90 м³)",
  "20t_120": "Сцепка 20 тонн (120 м³)",

  "10 тонн": "10 тонн",
  "20 тонн (82 м³)": "20 тонн (82 м³)",
  "20 тонн (90 м³)": "20 тонн (90 м³)",
  "Сцепка 20 тонн (120 м³)": "Сцепка 20 тонн (120 м³)"
};

  function showLoading(isLoading) {
    loading.classList.toggle("hidden", !isLoading);
    submitBtn.disabled = isLoading;
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
  }

  function hideError() {
    errorBox.textContent = "";
    errorBox.classList.add("hidden");
  }

  function showResult() {
    resultCard.classList.remove("hidden");
    placeholderCard.classList.add("hidden");
  }

  function resetResult() {
    routeEl.textContent = "—";
    officeEl.textContent = "—";
    predEl.textContent = "—";
    capacityEl.textContent = "—";
    allocatedEl.innerHTML = "—";
  }

  function formatPrediction(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "—";
    }
    return Number(value).toFixed(2);
  }

  function renderAllocatedVehicles(vehicles) {
  if (!vehicles || typeof vehicles !== "object") {
    allocatedEl.innerHTML = "—";
    return;
  }

  const filtered = Object.entries(vehicles).filter(([_, count]) => Number(count) > 0);

  if (filtered.length === 0) {
    allocatedEl.innerHTML = '<div class="fleet-empty">Подходящий транспорт не требуется</div>';
    return;
  }

  allocatedEl.innerHTML = "";

  filtered.forEach(([rawName, count]) => {
    const displayName = TRUCK_LABELS[rawName] || rawName;

    const row = document.createElement("div");
    row.className = "fleet-result-row";

    const nameEl = document.createElement("span");
    nameEl.className = "fleet-result-name";
    nameEl.textContent = displayName;

    const valueEl = document.createElement("span");
    valueEl.className = "fleet-result-value";
    valueEl.textContent = `${count} шт.`;

    row.appendChild(nameEl);
    row.appendChild(valueEl);
    allocatedEl.appendChild(row);
  });
}

  function updateStatusBadge(vehiclesText) {
    if (!vehiclesText || vehiclesText === "—") {
      statusBadge.textContent = "Рекомендация по вызову транспорта: —";
      statusBadge.className = "status-badge neutral";
      return;
    }

    statusBadge.textContent = `Рекомендация по вызову транспорта: ${vehiclesText}`;
    statusBadge.className = "status-badge good";
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    hideError();
    resetResult();
    showLoading(true);

    const payload = {
      route_id: Number(document.getElementById("route_id").value),
      timestamp: document.getElementById("timestamp").value,
      fleet_10t: Number(document.getElementById("fleet_10t").value || 0),
      fleet_20t_82: Number(document.getElementById("fleet_20t_82").value || 0),
      fleet_20t_90: Number(document.getElementById("fleet_20t_90").value || 0),
      fleet_20t_120: Number(document.getElementById("fleet_20t_120").value || 0)
    };

    try {
      const response = await fetch("/predict-point", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      let data;
      try {
        data = await response.json();
      } catch {
        throw new Error("Сервис вернул некорректный ответ.");
      }

      if (!response.ok) {
        throw new Error(data.detail || data.error || "Не удалось получить прогноз.");
      }

      routeEl.textContent = data.route_id ?? "—";
      officeEl.textContent = data.office_from_id ?? "—";
      predEl.textContent = formatPrediction(data.predicted_target_2h);
      capacityEl.textContent = data.required_capacity ?? "—";

      const vehiclesText = renderAllocatedVehicles(data.allocated_vehicles);
      showResult();
    } catch (error) {
      showError(error.message || "Произошла ошибка при выполнении запроса.");
    } finally {
      showLoading(false);
    }
  });
});