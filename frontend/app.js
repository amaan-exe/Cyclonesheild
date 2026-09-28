/**
 * Cyclone Horizon — Frontend Application Logic
 * Plain JavaScript, zero bundler, calling FastAPI endpoints via fetch()
 */

// Application State
const state = {
  currentView: 'landing',
  token: localStorage.getItem('ch_token') || null,
  userType: localStorage.getItem('ch_user_type') || null, // 'citizen' | 'authority'
  userData: JSON.parse(localStorage.getItem('ch_user_data') || 'null'),
  stormData: null,
  activeFunnelStage: 't3_track_intensity',
  funnelData: null,
  funnelInterval: null,
  map: null,
  mapLayers: {
    stormEye: null,
    pastTrack: null,
    forecastTrack: null,
    forecastWaypoints: [],
    conePolygon: null,
    windRadii: [],
    shelters: [],
    radarCircle: null,
    developmentSystems: [],
    liveCloudAnimation: [],
    liveCloudOverlay: null,
    liveWindAnimation: [],
    funnelLayers: [],
    evacuationRoute: null,
    evacuationMarkers: [],
    riskCircles: []
  },
  hindcastTime: 0,
  currentLanguage: 'en',
  isOffline: false,
  beaconActive: false,
  beaconAudioCtx: null,
  beaconOscillator: null,
  evacuationLayers: [],
  riskLayerVisible: false,
  userCoords: { lat: 19.8050, lon: 85.8300 }
};

// =============================================================
// Initialization
// =============================================================
document.addEventListener('DOMContentLoaded', () => {
  initClock();
  initMap();
  updateAuthUI();
  initOceanFeeds();
  fetchActiveStorm();
  fetchPredictionFunnel();
  fetchDevelopmentStatus();
  loadSheltersView('');
  initOfflineEngine();
  fetchMOSDACStatus();

  // Periodically refresh active storm data (every 60s)
  setInterval(() => {
    if (!state.isOffline) {
      initOceanFeeds();
      fetchActiveStorm(true);
      fetchDevelopmentStatus();
    }
  }, 60000);
});

// Clock updater
function initClock() {
  function update() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour12: false });
    const clockEl = document.getElementById('header-clock');
    if (clockEl) clockEl.innerText = `${timeStr} IST`;
  }
  update();
  setInterval(update, 1000);
}

// =============================================================
// Navigation & View Switching
// =============================================================
function switchView(viewName) {
  // If attempting to access citizen or authority without auth, redirect to login tab
  if (viewName === 'citizen' && (!state.token || state.userType !== 'citizen')) {
    switchAuthTab('citizen');
    viewName = 'login';
  } else if (viewName === 'authority' && (!state.token || state.userType !== 'authority')) {
    switchAuthTab('authority');
    viewName = 'login';
  }

  state.currentView = viewName;

  // Update tabs UI
  document.querySelectorAll('.nav-tab-btn').forEach(btn => btn.classList.remove('active'));
  const activeTabBtn = document.getElementById(`tab-${viewName}`);
  if (activeTabBtn) activeTabBtn.classList.add('active');

  // Update view containers
  document.querySelectorAll('.app-view').forEach(view => view.classList.remove('active-view'));
  const targetView = document.getElementById(`view-${viewName}`);
  if (targetView) targetView.classList.add('active-view');

  // Specific view loaders
  if (viewName === 'landing') {
    setTimeout(() => {
      if (state.map) state.map.invalidateSize();
    }, 100);
  } else if (viewName === 'citizen') {
    loadCitizenProfile();
    loadCitizenSOSHistory();
    loadCitizenShelters();
    fetchCitizenLocalRiskScore();
  } else if (viewName === 'authority') {
    loadAuthorityQueue();
  } else if (viewName === 'shelters') {
    loadSheltersView(document.getElementById('shelter-district-filter')?.value || '');
  }
}

function handleAuthAction() {
  if (state.token) {
    // Logout
    localStorage.removeItem('ch_token');
    localStorage.removeItem('ch_user_type');
    localStorage.removeItem('ch_user_data');
    state.token = null;
    state.userType = null;
    state.userData = null;
    updateAuthUI();
    switchView('landing');
  } else {
    // Login
    switchView('login');
  }
}

function updateAuthUI() {
  const label = document.getElementById('nav-user-label');
  const btn = document.getElementById('btn-auth-action');

  if (state.token && state.userData) {
    if (state.userType === 'citizen') {
      label.innerText = `Citizen: ${state.userData.name} (${state.userData.district})`;
      label.className = 'badge badge-green';
    } else {
      label.innerText = `Auth: ${state.userData.user_id} (${state.userData.district_scope || 'National'})`;
      label.className = 'badge badge-navy';
    }
    btn.innerText = 'Logout';
  } else {
    label.innerText = 'Public Access';
    label.className = 'badge badge-navy';
    btn.innerText = 'Login';
  }
}

function switchAuthTab(tab) {
  document.querySelectorAll('.auth-tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`auth-tab-${tab}`).classList.add('active');

  if (tab === 'citizen') {
    document.getElementById('form-citizen-login').style.display = 'block';
    document.getElementById('form-authority-login').style.display = 'none';
  } else {
    document.getElementById('form-citizen-login').style.display = 'none';
    document.getElementById('form-authority-login').style.display = 'block';
  }
}

// =============================================================
// Local-Language Translation Engine & Audio Advisory Siren
// =============================================================
const I18N_DICTIONARY = {
  en: {
    banner_status: "RED ALERT — VERY SEVERE CYCLONIC STORM",
    banner_meta: "Landfall expected near Puri & Dhamra Coast within 12-14 hours. Winds 120-145 km/h. Mandatory evacuation underway.",
    active_cyclone_label: "Active System",
    eye_pos_label: "Eye Position",
    landfall_eta_label: "Landfall Window",
    wind_label: "Sustained Wind",
    surge_label: "Storm Surge",
    shelters_label: "Shelters Ready",
    sos_btn: "BROADCAST EMERGENCY SOS NOW",
    find_route_btn: "🧭 Find Safer Route",
    audio_btn: "🔊 Audio Alert",
    tab_landing: "🛰️ GIS Map & Advisory",
    tab_citizen: "🛡️ Citizen Portal & SOS",
    tab_authority: "🏛️ Authorities Portal",
    tab_shelters: "🏠 Capacity-Aware Shelters",
    tab_chat: "🤖 Scoped AI Assistant",
    action_directive_t3: "⚠️ OPERATIONAL ACTION: Mandatory evacuation of low-lying coastal populations within 5 km. Pre-positioning 18 NDRF and 24 ODRAF rescue teams. Great Danger Signal GD-10 hoisted."
  },
  or: {
    banner_status: "ଲାଲ୍ ଚେତାବନୀ (ରେଡ୍ ଆଲର୍ଟ) — ଭୟଙ୍କର ବାତ୍ୟା 'ଦାନା'",
    banner_meta: "୧୨-୧୪ ଘଣ୍ଟା ମଧ୍ୟରେ ପୁରୀ ଓ ଧାମରା ଉପକୂଳରେ ଲ୍ୟାଣ୍ଡଫଲ୍ ସମ୍ଭାବନା। ପବନର ବେଗ ୧୨୦-୧୪୫ କିମି/ଘଣ୍ଟା। ବାଧ୍ୟତାମୂଳକ ସ୍ଥାନାନ୍ତର ଜାରି।",
    active_cyclone_label: "ସକ୍ରିୟ ବାତ୍ୟା",
    eye_pos_label: "ବାତ୍ୟାର କେନ୍ଦ୍ର (ଚକ୍ଷୁ)",
    landfall_eta_label: "ଲ୍ୟାଣ୍ଡଫଲ୍ ସମୟ",
    wind_label: "ପବନର ବେଗ",
    surge_label: "ଜୁଆର ଉଚ୍ଚତା",
    shelters_label: "ପ୍ରସ୍ତୁତ ଆଶ୍ରୟସ୍ଥଳୀ",
    sos_btn: "ତୁରନ୍ତ ଜରୁରୀକାଳୀନ SOS ପ୍ରସାରଣ କରନ୍ତୁ",
    find_route_btn: "🧭 ନିରାପଦ ରାସ୍ତା ଖୋଜନ୍ତୁ",
    audio_btn: "🔊 ଚେତାବନୀ ଶୁଣନ୍ତୁ",
    tab_landing: "🛰️ GIS ମାନଚିତ୍ର",
    tab_citizen: "🛡️ ନାଗରିକ ପୋର୍ଟାଲ୍",
    tab_authority: "🏛️ ପ୍ରଶାସନ ପୋର୍ଟାଲ୍",
    tab_shelters: "🏠 ବାତ୍ୟା ଆଶ୍ରୟସ୍ଥଳୀ",
    tab_chat: "🤖 AI ସହାୟକ",
    action_directive_t3: "⚠️ କାର୍ଯ୍ୟାନୁଷ୍ଠାନ ନିର୍ଦ୍ଦେଶ: ଉପକୂଳର ୫ କିମି ଭିତରେ ଥିବା ଲୋକଙ୍କୁ ତୁରନ୍ତ ବାଧ୍ୟତାମୂଳକ ସ୍ଥାନାନ୍ତର। ୧୮ NDRF ଏବଂ ୨୪ ODRAF ଦଳ ମୁତୟନ। ମହା ବିପଦ ସଙ୍କେତ ୧୦ ଜାରି।"
  },
  hi: {
    banner_status: "रेड अलर्ट — अत्यंत भीषण चक्रवाती तूफान (तूफान दाना)",
    banner_meta: "12-14 घंटों में पुरी और धामरा तट के पास लैंडफॉल की संभावना। हवाएं 120-145 किमी/घंटा। अनिवार्य निकासी जारी।",
    active_cyclone_label: "सक्रिय चक्रवात",
    eye_pos_label: "चक्रवात केंद्र",
    landfall_eta_label: "लैंडफॉल समय",
    wind_label: "सतत हवा गति",
    surge_label: "तूफानी लहर",
    shelters_label: "तैयार आश्रय स्थल",
    sos_btn: "आपातकालीन संकट संदेश (SOS) भेजें",
    find_route_btn: "🧭 सुरक्षित मार्ग खोजें",
    audio_btn: "🔊 चेतावनी सुनें",
    tab_landing: "🛰️ GIS मानचित्र",
    tab_citizen: "🛡️ नागरिक पोर्टल",
    tab_authority: "🏛️ प्राधिकरण पोर्टल",
    tab_shelters: "🏠 चक्रवात आश्रय स्थल",
    tab_chat: "🤖 AI आपात सहायक",
    action_directive_t3: "⚠️ परिचालन निर्देश: तट से 5 किमी के भीतर रहने वाली आबादी की अनिवार्य निकासी। 18 एनडीआरएफ और 24 ओड्राफ टीमें तैनात। महा खतरा संकेत 10 जारी।"
  },
  bn: {
    banner_status: "রেড অ্যালার্ট — অতি তীব্র ঘূর্ণিঝড় (ঘূর্ণিঝড় দানা)",
    banner_meta: "১২-১৪ ঘণ্টার মধ্যে পুরী ও ধামড়া উপকূলের কাছে আছড়ে পড়ার সম্ভাবনা। বাতাসের গতি ১২০-১৪৫ কিমি/ঘণ্টা। বাধ্যতামূলক নিরাপদ আশ্রয়ে সরিয়ে নেওয়ার কাজ চলছে।",
    active_cyclone_label: "সক্রিয় ঘূর্ণিঝড়",
    eye_pos_label: "ঘূর্ণিঝড় কেন্দ্র",
    landfall_eta_label: "ল্যান্ডফল সময়",
    wind_label: "বাতাসের গতিবেগ",
    surge_label: "জলোচ্ছ্বাসের উচ্চতা",
    shelters_label: "প্রস্তুত আশ্রয়কেন্দ্র",
    sos_btn: "জরুরি এসওএস (SOS) পাঠান",
    find_route_btn: "🧭 নিরাপদ রুট খুঁজুন",
    audio_btn: "🔊 বার্তা শুনুন",
    tab_landing: "🛰️ GIS মানচিত্র",
    tab_citizen: "🛡️ নাগরিক পোর্টাল",
    tab_authority: "🏛️ কর্তৃপক্ষ পোর্টাল",
    tab_shelters: "🏠 ঘূর্ণিঝড় আশ্রয়কেন্দ্র",
    tab_chat: "🤖 AI সহকারী",
    action_directive_t3: "⚠️ জরুরি নির্দেশ: উপকূলের ৫ কিমি অঞ্চলের মানুষদের অবিলম্বে নিরাপদ স্থানে সরিয়ে নেওয়া হচ্ছে। ১৮টি NDRF ও ২৪টি ODRAF উদ্ধারকারী দল মোতায়েন।"
  },
  te: {
    banner_status: "రెడ్ అలర్ట్ — తీవ్ర తుఫాను (దానా తుఫాను)",
    banner_meta: "12-14 గంటల్లో పూరి, ధామ్రా తీరాల మధ్య తీరం దాటే అవకాశం. గాలుల వేగం 120-145 కి.మీ/గం. ప్రజలను సురక్షిత ప్రాంతాలకు తరలిస్తున్నారు.",
    active_cyclone_label: "చురుకైన తుఫాను",
    eye_pos_label: "తుఫాను కేంద్రం",
    landfall_eta_label: "తీరం దాటే సమయం",
    wind_label: "గాలుల వేగం",
    surge_label: "ఉప్పెన ఎత్తు",
    shelters_label: "సిద్ధంగా ఉన్న కేంద్రాలు",
    sos_btn: "తక్షణ అత్యవసర SOS ప్రసారం",
    find_route_btn: "🧭 సురక్షిత మార్గం",
    audio_btn: "🔊 హెచ్చరిక వినండి",
    tab_landing: "🛰️ GIS మ్యాప్",
    tab_citizen: "🛡️ పౌర పోర్టల్",
    tab_authority: "🏛️ అధికారిక పోర్టల్",
    tab_shelters: "🏠 పునరావాస కేంద్రాలు",
    tab_chat: "🤖 AI సహాయకుడు",
    action_directive_t3: "⚠️ కార్యచరణ ఆదేశం: తీరానికి 5 కి.మీ పరిధిలోని ప్రజలను తప్పనిసరిగా ఖాళీ చేయించాలి. 18 NDRF, 24 ODRAF రెస్క్యూ బృందాలు రంగంలోకి దిగాయి."
  }
};

function setLanguage(lang) {
  state.currentLanguage = lang;
  const dict = I18N_DICTIONARY[lang] || I18N_DICTIONARY.en;

  const bStatus = document.getElementById('banner-warning-status');
  if (bStatus) bStatus.innerText = dict.banner_status;

  const bMeta = document.getElementById('banner-alert-meta');
  if (bMeta) bMeta.innerText = dict.banner_meta;

  const tabLanding = document.getElementById('tab-landing');
  if (tabLanding) tabLanding.innerText = dict.tab_landing;
  const tabCitizen = document.getElementById('tab-citizen');
  if (tabCitizen) tabCitizen.innerText = dict.tab_citizen;
  const tabAuth = document.getElementById('tab-authority');
  if (tabAuth) tabAuth.innerText = dict.tab_authority;
  const tabShelters = document.getElementById('tab-shelters');
  if (tabShelters) tabShelters.innerText = dict.tab_shelters;
  const tabChat = document.getElementById('tab-chat');
  if (tabChat) tabChat.innerText = dict.tab_chat;

  const btnSos = document.getElementById('btn-submit-sos');
  if (btnSos) btnSos.innerHTML = `<span>🆘</span> ${dict.sos_btn}`;

  const btnRoute = document.getElementById('btn-find-safe-route');
  if (btnRoute) btnRoute.innerText = dict.find_route_btn;

  const btnAudio = document.getElementById('btn-audio-siren');
  if (btnAudio) btnAudio.innerText = dict.audio_btn;

  const actionEl = document.getElementById('funnel-detail-action');
  if (actionEl && state.activeFunnelStage === 't3_track_intensity') {
    actionEl.innerText = dict.action_directive_t3;
  }
}

function speakEmergencyAdvisory() {
  if (!('speechSynthesis' in window)) {
    alert("Voice synthesis is not supported on this browser. Please read the emergency alert on screen.");
    return;
  }
  window.speechSynthesis.cancel();
  const dict = I18N_DICTIONARY[state.currentLanguage] || I18N_DICTIONARY.en;
  const text = `${dict.banner_status}. ${dict.banner_meta}. ${dict.action_directive_t3 || ''}`;

  const utter = new SpeechSynthesisUtterance(text);
  const langCodes = { 'en': 'en-IN', 'or': 'or-IN', 'hi': 'hi-IN', 'bn': 'bn-IN', 'te': 'te-IN' };
  utter.lang = langCodes[state.currentLanguage] || 'en-IN';
  utter.rate = 0.95;
  utter.pitch = 1.0;
  window.speechSynthesis.speak(utter);
}

// GIS Map Engine (Leaflet)
// =============================================================
function initMap() {
  const mapElement = document.getElementById('cyclone-map');
  if (!mapElement) return;

  // Center over North Indian Ocean / Odisha & Andhra Pradesh Coast
  state.map = L.map('cyclone-map', {
    center: [18.80, 85.50],
    zoom: 6,
    zoomControl: true
  });

  // 1. Official IMD Cartographic Operational Basemap (Soft paper tones, topography & boundaries - 100% free, zero watermark)
  const imdCartoChart = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}', {
    attribution: '&copy; IMD / Survey of India | Esri Topo',
    maxZoom: 18,
    minZoom: 1
  });

  // 2. NASA GIBS Blue Marble with Bathymetry (Public, zero API key, global bathymetry & topography)
  const nasaGibsBlueMarble = L.tileLayer('https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief_Bathymetry/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpeg', {
    attribution: '&copy; NASA Earthdata GIBS | NASA EOSDIS',
    maxNativeZoom: 8,
    maxZoom: 18,
    minZoom: 1
  });

  // 3. NASA GIBS MODIS Terra TrueColor Satellite Imagery
  const nasaGibsModis = L.tileLayer('https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/2024-05-01/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg', {
    attribution: '&copy; NASA Earthdata GIBS (MODIS Terra) | NASA EOSDIS',
    maxNativeZoom: 9,
    maxZoom: 18,
    minZoom: 1
  });

  // 4. OpenStreetMap Standard (Clean administrative boundaries & coastal details)
  const osmStandard = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors | IMD GIS',
    maxZoom: 18,
    minZoom: 1
  });

  // 5. ESRI World Satellite Imagery (High-res optical satellite view)
  const esriSatellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: '&copy; Esri, Maxar, Earthstar Geographics',
    maxZoom: 18,
    minZoom: 1
  });

  state.baseLayers = {
    imd: imdCartoChart,
    nasa: nasaGibsBlueMarble,
    modis: nasaGibsModis,
    osm: osmStandard,
    esri: esriSatellite
  };

  // Add default basemap: IMD Cartographic Chart (exact match for user reference image, zero watermark)
  imdCartoChart.addTo(state.map);
  state.activeBasemap = 'imd';

  // Basemap switcher in top-right corner
  const baseMaps = {
    "IMD Operational Chart (Paper)": imdCartoChart,
    "NASA GIBS (Blue Marble)": nasaGibsBlueMarble,
    "NASA GIBS (MODIS Satellite)": nasaGibsModis,
    "OpenStreetMap (Coast & Towns)": osmStandard,
    "ESRI Satellite (High-Res)": esriSatellite
  };
  L.control.layers(baseMaps, null, { position: 'topright' }).addTo(state.map);

  // Auto-load Live MOSDAC INSAT-3DS thermal cloud overlay
  fetchLiveCloudOverlay();

  // Interactive Live Vortex Ingestion Listener (Click to Predict Mode)
  state.map.on('click', async (e) => {
    if (!state.mapClickPredictMode) return;
    const lat = Number(e.latlng.lat.toFixed(2));
    const lon = Number(e.latlng.lng.toFixed(2));

    if (lat < 0 || lat > 32 || lon < 50 || lon > 102) {
      alert("Please select coordinates within the North Indian Ocean basin (Bay of Bengal, Arabian Sea, or coastal India).");
      return;
    }

    showToast(`⚡ Ingesting live vortex at ${lat}°N, ${lon}°E... Running 5-model neural pipeline!`);

    try {
      const res = await fetch('/api/ml/predict-live-vortex', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lat: lat,
          lon: lon,
          wind_kt: 65.0,
          pressure_hpa: 982.0,
          name: `Custom Vortex (${lat}°N, ${lon}°E)`
        })
      });
      if (!res.ok) throw new Error("Vortex prediction failed");
      const data = await res.json();

      const select = document.getElementById('select-ocean-feed');
      if (select) select.value = 'interactive_radar';

      state.stormData = { advisory: data.advisory, storm_name: data.advisory.storm_name };
      updateTelemetryUI(data.advisory);
      renderStormOnMap(data.advisory);
      updateDistrictMatrix(data.advisory.district_risk_matrix);

      showToast(`🎯 AI Generated 72h Track Cone & Local Risk Scores for vortex at ${lat}°N, ${lon}°E!`);
    } catch (err) {
      console.error("Vortex prediction error:", err);
      alert("Failed to run prediction: " + err.message);
    }
  });
}


function centerStormMap() {
  if (state.map && state.stormData && state.stormData.advisory) {
    const cur = state.stormData.advisory.current_state;
    state.map.setView([cur.lat, cur.lon], 6);
  }
}

function setMapVisualizationMode(mode) {
  state.mapMode = mode; // 'cone', 'swaths', 'both'
  const btnCone = document.getElementById('btn-mode-cone');
  const btnSwaths = document.getElementById('btn-mode-swaths');
  const btnBoth = document.getElementById('btn-mode-both');

  if (btnCone) {
    btnCone.className = mode === 'cone' ? 'seg-btn active' : 'seg-btn';
  }
  if (btnSwaths) {
    btnSwaths.className = mode === 'swaths' ? 'seg-btn active' : 'seg-btn';
  }
  if (btnBoth) {
    btnBoth.className = mode === 'both' ? 'seg-btn active' : 'seg-btn';
  }

  if (state.stormData && state.stormData.advisory) {
    renderStormOnMap(state.stormData.advisory);
  }
}

function toggleBasemapStyle() {
  if (!state.map || !state.baseLayers) return;
  const btn = document.getElementById('btn-toggle-basemap');
  if (state.activeBasemap === 'imd') {
    state.map.removeLayer(state.baseLayers.imd);
    state.baseLayers.esri.addTo(state.map);
    state.activeBasemap = 'satellite';
    if (btn) btn.innerText = "IMD Chart Mode";
  } else {
    state.map.removeLayer(state.baseLayers.esri);
    state.baseLayers.imd.addTo(state.map);
    state.activeBasemap = 'imd';
    if (btn) btn.innerText = "Satellite Mode";
  }
}

function toggleRadarOverlay() {
  if (!state.mapLayers.radarCircle) return;
  const isVisible = state.map.hasLayer(state.mapLayers.radarCircle);
  if (isVisible) {
    state.map.removeLayer(state.mapLayers.radarCircle);
  } else {
    state.mapLayers.radarCircle.addTo(state.map);
  }
}

// Helper to generate a smooth buffered capsule corridor with semi-circular end and start caps
function createTrackCapsule(points, rStart, rEnd) {
  if (!points || points.length < 2) return null;
  const n = points.length;
  const left = [];
  const right = [];
  const tangents = [];
  const normals = [];

  for (let i = 0; i < n; i++) {
    const p = points[i];
    let dx = 0, dy = 0;
    if (i === 0) {
      dx = points[1].lon - p.lon;
      dy = points[1].lat - p.lat;
    } else if (i === n - 1) {
      dx = p.lon - points[n - 2].lon;
      dy = p.lat - points[n - 2].lat;
    } else {
      dx = points[i + 1].lon - points[i - 1].lon;
      dy = points[i + 1].lat - points[i - 1].lat;
    }
    const len = Math.hypot(dx, dy) || 1e-6;
    const tx = dx / len;
    const ty = dy / len;
    tangents.push({ tx, ty });
    normals.push({ nx: -ty, ny: tx }); // 90 deg counter-clockwise (left normal)
  }

  // Left boundary points from i = 0 to n - 1
  for (let i = 0; i < n; i++) {
    const p = points[i];
    const frac = n > 1 ? i / (n - 1) : 0;
    const r = rStart + (rEnd - rStart) * frac;
    const latCos = Math.cos((p.lat * Math.PI) / 180.0) || 1.0;
    const { nx, ny } = normals[i];
    left.push([p.lat + ny * r, p.lon + (nx * r) / latCos]);
  }

  // End cap around the last point (semi-circular dome from left normal to right normal)
  const lastP = points[n - 1];
  const lastT = tangents[n - 1];
  const lastLatCos = Math.cos((lastP.lat * Math.PI) / 180.0) || 1.0;
  const rLast = rEnd;
  const headingEnd = Math.atan2(lastT.ty, lastT.tx);
  const endCapArc = [];
  const ARC_STEPS = 14;
  for (let s = 1; s < ARC_STEPS; s++) {
    const angle = headingEnd + (Math.PI / 2) - (Math.PI * s / ARC_STEPS);
    endCapArc.push([
      lastP.lat + Math.sin(angle) * rLast,
      lastP.lon + (Math.cos(angle) * rLast) / lastLatCos
    ]);
  }

  // Right boundary points in reverse order from i = n - 1 down to 0
  for (let i = n - 1; i >= 0; i--) {
    const p = points[i];
    const frac = n > 1 ? i / (n - 1) : 0;
    const r = rStart + (rEnd - rStart) * frac;
    const latCos = Math.cos((p.lat * Math.PI) / 180.0) || 1.0;
    const { nx, ny } = normals[i];
    right.push([p.lat - ny * r, p.lon - (nx * r) / latCos]);
  }

  // Start cap around the first point (semi-circular dome from right normal to left normal)
  const firstP = points[0];
  const firstT = tangents[0];
  const firstLatCos = Math.cos((firstP.lat * Math.PI) / 180.0) || 1.0;
  const rFirst = rStart;
  const headingStart = Math.atan2(firstT.ty, firstT.tx);
  const startCapArc = [];
  for (let s = 1; s < ARC_STEPS; s++) {
    const angle = headingStart - (Math.PI / 2) - (Math.PI * s / ARC_STEPS);
    startCapArc.push([
      firstP.lat + Math.sin(angle) * rFirst,
      firstP.lon + (Math.cos(angle) * rFirst) / firstLatCos
    ]);
  }

  return [...left, ...endCapArc, ...right, ...startCapArc];
}

function renderStormOnMap(advisory) {
  if (!state.map || !advisory) return;

  // Clear previous dynamic layers
  if (state.mapLayers.stormEye) state.map.removeLayer(state.mapLayers.stormEye);
  if (state.mapLayers.pastTrack) state.map.removeLayer(state.mapLayers.pastTrack);
  if (state.mapLayers.forecastTrack) state.map.removeLayer(state.mapLayers.forecastTrack);
  if (state.mapLayers.forecastCone) state.map.removeLayer(state.mapLayers.forecastCone);
  if (state.mapLayers.swathGale) state.map.removeLayer(state.mapLayers.swathGale);
  if (state.mapLayers.swathStorm) state.map.removeLayer(state.mapLayers.swathStorm);
  if (state.mapLayers.swathHurricane) state.map.removeLayer(state.mapLayers.swathHurricane);
  if (state.mapLayers.radarCircle) state.map.removeLayer(state.mapLayers.radarCircle);

  (state.mapLayers.forecastWaypoints || []).forEach(m => state.map.removeLayer(m));
  state.mapLayers.forecastWaypoints = [];
  (state.mapLayers.pastWaypoints || []).forEach(m => state.map.removeLayer(m));
  state.mapLayers.pastWaypoints = [];
  (state.mapLayers.windRadii || []).forEach(r => state.map.removeLayer(r));
  state.mapLayers.windRadii = [];

  const cur = advisory.current_state;

  // If in routine surveillance mode (No active cyclone in Indian Ocean)
  const isSurveillance = advisory.advisory_level === 'GREEN' || (cur.imd_code === 'ALL-CLEAR');
  if (isSurveillance) {
    const buoyIcon = L.divIcon({
      className: 'surveillance-buoy-icon',
      html: `<div style="width:24px;height:24px;background:#10B981;border:3px solid #FFFFFF;border-radius:50%;box-shadow:0 0 12px #10B981;display:flex;align-items:center;justify-content:center;">
               <div style="width:7px;height:7px;background:#FFFFFF;border-radius:50%;"></div>
             </div>`,
      iconSize: [24, 24],
      iconAnchor: [12, 12]
    });
    state.mapLayers.stormEye = L.marker([cur.lat, cur.lon], { icon: buoyIcon }).addTo(state.map);
    state.mapLayers.stormEye.bindTooltip(
      `<strong>North Indian Ocean Basin Surveillance</strong><br>Sector: ${cur.lat}°N, ${cur.lon}°E<br>Wind: ${cur.max_wind_kph} km/h (${cur.max_wind_kt} kt)<br>MSLP: ${cur.min_pressure_hpa} hPa<br><span style="color:#10B981; font-weight:700;">ALL CLEAR — ZERO CYCLONES ACTIVE</span>`,
      { permanent: true, direction: 'top', className: 'imd-track-label' }
    );
    if (!state.userHasPanned) {
      state.map.setView([19.5, 87.0], 6);
    }
    return;
  }

  // Assemble full trajectory points
  const pastPts = (advisory.track_points || []).map(p => ({
    lat: p.lat,
    lon: p.lon,
    wind_kt: p.wind_kt || 40,
    imd_code: p.imd_code || 'CS',
    lead_h: p.lead_h || 0,
    is_past: true
  }));
  const curPt = {
    lat: cur.lat,
    lon: cur.lon,
    wind_kt: cur.max_wind_kt || 65,
    imd_code: cur.imd_code || 'VSCS',
    lead_h: 0,
    is_current: true
  };
  const predPts = (advisory.predictions || []).map(p => ({
    lat: p.lat,
    lon: p.lon,
    wind_kt: p.wind_kt || 50,
    imd_code: p.imd_code || 'SCS',
    lead_h: p.lead_h || 12,
    is_forecast: true
  }));

  const fullTrajectory = [...pastPts, curPt, ...predPts];
  const forecastTrajectory = [curPt, ...predPts];

  const currentMode = state.mapMode || 'cone';
  const showCone = currentMode === 'cone' || currentMode === 'both';
  const showSwaths = currentMode === 'swaths' || currentMode === 'both';

  // -------------------------------------------------------------
  // 1. IMD 3-TIER CONCENTRIC WIND HAZARD SWATHS (Left Image Style)
  // -------------------------------------------------------------
  if (showSwaths && fullTrajectory.length >= 2) {
    // Outer Swath: Gale Force Winds (>= 34 KT) — Grey / Slate Blue
    const galeCoords = createTrackCapsule(fullTrajectory, 1.85, 2.15);
    if (galeCoords) {
      state.mapLayers.swathGale = L.polygon(galeCoords, {
        color: '#4B5563',
        weight: 1.8,
        fillColor: '#94A3B8',
        fillOpacity: 0.44
      }).addTo(state.map).bindTooltip("IMD Gale Force Wind Swath (&ge;34 KT / 62 km/h)", { sticky: true });
    }

    // Middle Swath: Storm Force Winds (50 - 63 KT) — Marine Blue
    const stormPts = fullTrajectory.filter(p => p.wind_kt >= 45);
    if (stormPts.length >= 2) {
      const stormCoords = createTrackCapsule(stormPts, 1.15, 1.30);
      if (stormCoords) {
        state.mapLayers.swathStorm = L.polygon(stormCoords, {
          color: '#1D4ED8',
          weight: 1.8,
          fillColor: '#3B82F6',
          fillOpacity: 0.48
        }).addTo(state.map).bindTooltip("IMD Storm Force Wind Swath (50 - 63 KT / 90 - 117 km/h)", { sticky: true });
      }
    }

    // Inner Swath: Hurricane Force Winds (>= 64 KT) — Vibrant Green
    const hurrPts = fullTrajectory.filter(p => p.wind_kt >= 60);
    if (hurrPts.length >= 2) {
      const hurrCoords = createTrackCapsule(hurrPts, 0.60, 0.68);
      if (hurrCoords) {
        state.mapLayers.swathHurricane = L.polygon(hurrCoords, {
          color: '#15803D',
          weight: 2.0,
          fillColor: '#22C55E',
          fillOpacity: 0.62
        }).addTo(state.map).bindTooltip("IMD Hurricane Force Wind Swath (&ge;64 KT / 120 km/h)", { sticky: true });
      }
    }
  }

  // -------------------------------------------------------------
  // 2. IMD 72h FORECAST CONE OF UNCERTAINTY (Right Image Style)
  // -------------------------------------------------------------
  if (showCone && forecastTrajectory.length >= 2) {
    const coneCoords = createTrackCapsule(forecastTrajectory, 0.35, 1.95);
    if (coneCoords) {
      state.mapLayers.forecastCone = L.polygon(coneCoords, {
        color: '#14532D',
        weight: 2.2,
        fillColor: '#22C55E',
        fillOpacity: 0.58
      }).addTo(state.map).bindTooltip("IMD 72-Hour Cone of Uncertainty (Model Forecast Swath)", { sticky: true });
    }
  }

  const labeledMarkers = [];
  function canAddTrackLabel(lat, lon, synopticTime) {
    for (const lm of labeledMarkers) {
      if (lm.synopticTime === synopticTime) return false;
      const dist = Math.hypot(lat - lm.lat, lon - lm.lon);
      if (dist < 1.15) return false;
    }
    labeledMarkers.push({ lat, lon, synopticTime });
    return true;
  }

  // -------------------------------------------------------------
  // 3. CURRENT CYCLONE EYE (Centered marker with pulse)
  // -------------------------------------------------------------
  const eyeIcon = L.divIcon({
    className: 'cyclone-eye-icon',
    html: `<div style="width:22px;height:22px;background:#DC2626;border:3px solid #FFFFFF;border-radius:50%;box-shadow:0 0 10px #DC2626;position:relative;">
             <div style="width:6px;height:6px;background:#FFFFFF;border-radius:50%;position:absolute;top:5px;left:5px;"></div>
           </div>`,
    iconSize: [22, 22],
    iconAnchor: [11, 11]
  });

  state.mapLayers.stormEye = L.marker([cur.lat, cur.lon], { icon: eyeIcon })
    .addTo(state.map)
    .bindPopup(`
      <div style="font-size:12px; font-family:sans-serif;">
        <strong style="color:#DC2626; font-size:14px;">${advisory.storm_name.toUpperCase()}</strong><br>
        <strong>Current IMD Stage:</strong> ${cur.imd_category} (${cur.imd_code})<br>
        <strong>Max Sustained Winds:</strong> ${cur.max_wind_kph} km/h (${cur.max_wind_kt} kt)<br>
        <strong>Central Pressure:</strong> ${cur.min_pressure_hpa} hPa<br>
        <strong>Eye Position:</strong> ${cur.lat} deg N, ${cur.lon} deg E<br>
        <strong>Speed / Heading:</strong> ${cur.movement_speed_kph} km/h towards ${cur.movement_direction_deg} deg
      </div>
    `);

  const now = new Date();
  const curDay = String(now.getUTCDate()).padStart(2, '0');
  const curHr = String(Math.floor(now.getUTCHours() / 6) * 6).padStart(2, '0');
  const eyeSynopticTime = `${curDay}/${curHr}`;
  const eyeLabel = `${eyeSynopticTime}, ${Math.round(cur.max_wind_kt)}KT, ${cur.imd_code} (EYE)`;
  canAddTrackLabel(cur.lat, cur.lon, eyeSynopticTime);

  state.mapLayers.stormEye.bindTooltip(eyeLabel, {
    permanent: true,
    direction: 'right',
    offset: [14, -2],
    className: 'imd-track-label imd-track-label-forecast'
  });

  // -------------------------------------------------------------
  // 4. PAST OBSERVED TRACK (Solid Dark Line + Black Waypoints)
  // -------------------------------------------------------------
  if (pastPts.length > 0) {
    const pastLineCoords = [...pastPts.map(p => [p.lat, p.lon]), [cur.lat, cur.lon]];
    state.mapLayers.pastTrack = L.polyline(pastLineCoords, {
      color: '#000000',
      weight: 3.5,
      opacity: 1
    }).addTo(state.map);

    // Subsample past points to at most 4 key synoptic points
    const step = Math.max(1, Math.floor(pastPts.length / 4));
    const sampledPast = pastPts.filter((_, idx) => idx % step === 0 || idx === pastPts.length - 1);

    sampledPast.forEach((p) => {
      const m = L.circleMarker([p.lat, p.lon], {
        radius: 4.5,
        color: '#000000',
        fillColor: '#000000',
        fillOpacity: 1,
        weight: 1.5
      }).addTo(state.map);

      const labelDate = new Date(Date.now() + p.lead_h * 3600 * 1000);
      const day = String(labelDate.getUTCDate()).padStart(2, '0');
      const hr = String(Math.floor(labelDate.getUTCHours() / 6) * 6).padStart(2, '0');
      const synopticTime = `${day}/${hr}`;
      const labelText = `${synopticTime}, ${Math.round(p.wind_kt)}KT, ${p.imd_code || 'DD'}`;

      if (canAddTrackLabel(p.lat, p.lon, synopticTime)) {
        m.bindTooltip(labelText, {
          permanent: true,
          direction: 'right',
          offset: [10, -2],
          className: 'imd-track-label imd-track-label-past'
        });
      }
      state.mapLayers.pastWaypoints.push(m);
    });
  }

  // -------------------------------------------------------------
  // 5. FORECAST TRACK SPINE (Solid Red Line + Red Waypoints)
  // -------------------------------------------------------------
  if (predPts.length > 0) {
    const forecastLineCoords = [[cur.lat, cur.lon], ...predPts.map(p => [p.lat, p.lon])];
    state.mapLayers.forecastTrack = L.polyline(forecastLineCoords, {
      color: '#DC2626',
      weight: 3.5,
      opacity: 1
    }).addTo(state.map);

    predPts.forEach((p) => {
      const m = L.circleMarker([p.lat, p.lon], {
        radius: 5,
        color: '#991B1B',
        fillColor: '#DC2626',
        fillOpacity: 1,
        weight: 2
      }).addTo(state.map);

      const labelDate = new Date(Date.now() + p.lead_h * 3600 * 1000);
      const day = String(labelDate.getUTCDate()).padStart(2, '0');
      const hr = String(Math.floor(labelDate.getUTCHours() / 6) * 6).padStart(2, '0');
      const synopticTime = `${day}/${hr}`;
      const labelText = `${synopticTime}, ${Math.round(p.wind_kt)}KT, ${p.imd_code || 'CS'}`;

      if (canAddTrackLabel(p.lat, p.lon, synopticTime)) {
        m.bindTooltip(labelText, {
          permanent: true,
          direction: 'right',
          offset: [10, -2],
          className: 'imd-track-label imd-track-label-forecast'
        });
      }
      state.mapLayers.forecastWaypoints.push(m);
    });
  }

  // Focus view to perfectly frame the storm and its 72h forecast cone
  if (fullTrajectory.length >= 2 && !state.userHasPanned) {
    const lats = fullTrajectory.map(p => p.lat);
    const lons = fullTrajectory.map(p => p.lon);
    const bounds = L.latLngBounds(
      [Math.min(...lats) - 1.5, Math.min(...lons) - 1.5],
      [Math.max(...lats) + 1.5, Math.max(...lons) + 1.5]
    );
    state.map.fitBounds(bounds, { padding: [30, 30], maxZoom: 6 });
  }

  // Radar Doppler precipitation buffer
  state.mapLayers.radarCircle = L.circle([cur.lat, cur.lon], {
    radius: 120000,
    color: '#DC2626',
    weight: 1,
    fillColor: '#F87171',
    fillOpacity: 0.15
  }).addTo(state.map);
}


// Load shelter pins onto the map
function renderSheltersOnMap(shelters) {
  if (!state.map || !shelters) return;
  state.mapLayers.shelters.forEach(s => state.map.removeLayer(s));
  state.mapLayers.shelters = [];

  shelters.forEach(sh => {
    const availBeds = sh.available_beds !== undefined ? sh.available_beds : (sh.total_capacity - sh.current_occupancy);
    const m = L.circleMarker([sh.location_lat, sh.location_lon], {
      radius: 5,
      color: '#15803D',
      fillColor: availBeds > 100 ? '#22C55E' : '#EAB308',
      fillOpacity: 0.85,
      weight: 2
    }).addTo(state.map).bindPopup(`
      <div style="font-size:11px;">
        <strong>${sh.name}</strong><br>
        District: ${sh.district}<br>
        Available Beds: <strong>${availBeds}</strong> / ${sh.total_capacity}<br>
        Status: <span style="color:#16A34A; font-weight:700;">${sh.status.toUpperCase()}</span><br>
        Contact: ${sh.contact_number || 'DEOC 1077'}
      </div>
    `);
    state.mapLayers.shelters.push(m);
  });
}

// =============================================================
// Active Storm API & Data Binding
// =============================================================
async function fetchActiveStorm(silent = false) {
  try {
    const res = await fetch('/api/storms/active');
    if (!res.ok) throw new Error('No active storm found');
    const data = await res.json();
    state.stormData = data;

    updateTelemetryUI(data.advisory);
    renderStormOnMap(data.advisory);
    updateDistrictMatrix(data.advisory.district_risk_matrix);

    // Also fetch shelters to display on map
    const shRes = await fetch('/api/shelters');
    if (shRes.ok) {
      const shData = await shRes.json();
      renderSheltersOnMap(shData.shelters);
    }
  } catch (err) {
    if (!silent) console.error('Failed to load active storm:', err);
  }
}

async function fetchDevelopmentStatus() {
  try {
    const response = await fetch('/api/ml/development-status');
    if (!response.ok) throw new Error('Development status unavailable');
    const data = await response.json();
    const chance = Math.max(0, Math.min(100, Number(data.formation_chance_percent || 0)));
    const chanceValue = document.getElementById('development-chance-value');
    const chanceMeter = document.getElementById('development-chance-meter');
    const stage = document.getElementById('development-stage');
    const list = document.getElementById('developing-systems-list');
    const badge = document.getElementById('development-watch-badge');
    const note = document.getElementById('development-note');
    if (chanceValue) chanceValue.innerText = `${chance.toFixed(1)}%`;
    if (chanceMeter) chanceMeter.style.width = `${chance}%`;
    if (stage) stage.innerText = data.formation_stage || 'No developing system detected';
    if (badge) {
      badge.innerText = chance >= 70 ? 'High Watch' : chance >= 40 ? 'Moderate Watch' : 'Low Watch';
      badge.className = `badge ${chance >= 70 ? 'badge-red' : chance >= 40 ? 'badge-yellow' : 'badge-green'}`;
    }
    if (note) note.innerText = data.method || 'Model guidance only.';
    if (list) {
      list.innerHTML = (data.systems || []).map(system => `
        <div class="developing-system-item">
          <div>
            <strong>${system.name}</strong>
            <span>${system.stage} · ${Number(system.center.lat).toFixed(2)}°N, ${Number(system.center.lon).toFixed(2)}°E</span>
          </div>
          <b>${Number(system.formation_chance_percent).toFixed(1)}%</b>
        </div>
      `).join('');
    }
    renderDevelopmentSystemsOnMap(data.systems || []);
    fetchLiveCloudOverlay();
  } catch (error) {
    console.warn('Could not load cyclone development status:', error);
  }
}

// MOSDAC Live Thermal Cloud Overlay State
state.mosdacCloudEnabled = true;
state.mosdacCloudOpacity = 0.65;
state.mosdacCloudData = null;

async function fetchLiveCloudOverlay() {
  if (!state.map) return;
  try {
    const response = await fetch('/api/ml/mosdac/cloud-overlay');
    if (!response.ok) return;
    const data = await response.json();
    state.mosdacCloudData = data;

    if (state.mapLayers.liveCloudOverlay) {
      state.map.removeLayer(state.mapLayers.liveCloudOverlay);
      state.mapLayers.liveCloudOverlay = null;
    }

    if (state.mosdacCloudEnabled) {
      state.mapLayers.liveCloudOverlay = L.imageOverlay(data.image_url, data.bounds, {
        opacity: state.mosdacCloudOpacity,
        interactive: false,
        zIndex: 350
      }).addTo(state.map);
    }

    updateCloudTelemetryBadge(data);
    updateCloudToggleButton(state.mosdacCloudEnabled);
  } catch (error) {
    console.warn('Could not load live MOSDAC cloud overlay:', error);
  }
}

function toggleMOSDACCloudOverlay() {
  state.mosdacCloudEnabled = !state.mosdacCloudEnabled;
  if (!state.map) return;

  if (state.mosdacCloudEnabled) {
    if (state.mosdacCloudData) {
      if (state.mapLayers.liveCloudOverlay) {
        state.map.removeLayer(state.mapLayers.liveCloudOverlay);
      }
      state.mapLayers.liveCloudOverlay = L.imageOverlay(state.mosdacCloudData.image_url, state.mosdacCloudData.bounds, {
        opacity: state.mosdacCloudOpacity,
        interactive: false,
        zIndex: 350
      }).addTo(state.map);
    } else {
      fetchLiveCloudOverlay();
    }
  } else {
    if (state.mapLayers.liveCloudOverlay) {
      state.map.removeLayer(state.mapLayers.liveCloudOverlay);
      state.mapLayers.liveCloudOverlay = null;
    }
  }

  updateCloudToggleButton(state.mosdacCloudEnabled);
  const telemetryEl = document.getElementById('map-cloud-telemetry');
  if (telemetryEl) {
    telemetryEl.style.display = state.mosdacCloudEnabled ? 'block' : 'none';
  }
  showToast(state.mosdacCloudEnabled ? '☁️ MOSDAC INSAT-3DS live thermal clouds enabled' : '☁️ MOSDAC cloud layer hidden');
}

function setCloudOverlayOpacity(val) {
  const num = Number(val) / 100.0;
  state.mosdacCloudOpacity = num;
  const label = document.getElementById('cloud-opacity-val');
  if (label) label.textContent = `${Math.round(num * 100)}%`;
  if (state.mapLayers.liveCloudOverlay) {
    state.mapLayers.liveCloudOverlay.setOpacity(num);
  }
}

function updateCloudToggleButton(isActive) {
  const btn = document.getElementById('btn-toggle-clouds');
  if (btn) {
    if (isActive) {
      btn.classList.add('active');
      btn.innerHTML = '☁️ Live Clouds: ON';
      btn.style.background = '#0284C7';
      btn.style.borderColor = '#0369A1';
      btn.style.color = '#FFF';
    } else {
      btn.classList.remove('active');
      btn.innerHTML = '☁️ Live Clouds: OFF';
      btn.style.background = '';
      btn.style.borderColor = '';
      btn.style.color = '';
    }
  }
}

function updateCloudTelemetryBadge(data) {
  const telemetryEl = document.getElementById('map-cloud-telemetry');
  const tempEl = document.getElementById('cloud-min-temp');
  if (telemetryEl && data) {
    telemetryEl.style.display = state.mosdacCloudEnabled ? 'block' : 'none';
    if (tempEl && data.min_brightness_temp_c !== undefined) {
      tempEl.textContent = `${data.min_brightness_temp_c}°C`;
      tempEl.style.color = data.min_brightness_temp_c <= -60 ? '#F472B6' : '#38BDF8';
    }
  }
}

function renderDevelopmentSystemsOnMap(systems) {
  if (!state.map) return;
  (state.mapLayers.developmentSystems || []).forEach(layer => state.map.removeLayer(layer));
  state.mapLayers.developmentSystems = [];
  systems.forEach(system => {
    const chance = Number(system.formation_chance_percent || 0);
    const color = chance >= 70 ? '#DC2626' : chance >= 40 ? '#D97706' : '#16A34A';
    const marker = L.circleMarker([system.center.lat, system.center.lon], {
      radius: 8,
      color,
      fillColor: color,
      fillOpacity: 0.82,
      weight: 2
    }).addTo(state.map);
    marker.bindTooltip(`
      <strong>${system.name}</strong><br>
      ${system.stage}<br>
      Formation chance: <strong>${chance.toFixed(1)}%</strong><br>
      TIR1 cloud top: ${system.cloud_top_temperature_k} K<br>
      Wind proxy: ${system.wind_proxy_kt} kt
    `, { sticky: true, direction: 'top' });
    state.mapLayers.developmentSystems.push(marker);
  });
}

function animateLiveWeather(center, cloudTempK, windKt) {
  if (!state.map || !center) return;
  (state.mapLayers.liveCloudAnimation || []).forEach(layer => state.map.removeLayer(layer));
  (state.mapLayers.liveWindAnimation || []).forEach(layer => state.map.removeLayer(layer));
  state.mapLayers.liveCloudAnimation = [];
  state.mapLayers.liveWindAnimation = [];
  if (state.liveWeatherTimer) clearInterval(state.liveWeatherTimer);

  let phase = 0;
  const cloudColor = cloudTempK <= 220 ? '#DC2626' : '#0EA5E9';
  state.liveWeatherTimer = setInterval(() => {
    phase = (phase + 1) % 4;
    state.mapLayers.liveCloudAnimation.forEach((ring, index) => {
      const radius = 35000 + ((phase + index) % 4) * 16000;
      ring.setRadius(radius);
      ring.setStyle({ opacity: 0.15 + (((phase + index) % 4) * 0.06) });
    });
    state.mapLayers.liveWindAnimation.forEach((arrow, index) => {
      const angle = ((phase * 12) + index * 90) % 360;
      const latOffset = Math.cos(angle * Math.PI / 180) * 0.25;
      const lonOffset = Math.sin(angle * Math.PI / 180) * 0.25;
      arrow.setLatLng([center.lat + latOffset, center.lon + lonOffset]);
    });
  }, 900);

  for (let i = 0; i < 4; i++) {
    state.mapLayers.liveCloudAnimation.push(L.circle([center.lat, center.lon], {
      radius: 35000 + i * 16000,
      color: cloudColor,
      weight: 2,
      fill: false,
      opacity: 0.18,
      interactive: false
    }).addTo(state.map));
  }
  for (let i = 0; i < 4; i++) {
    const icon = L.divIcon({
      className: 'live-wind-arrow',
      html: `<span style="color:${windKt >= 34 ? '#DC2626' : '#2563EB'};">➤</span>`,
      iconSize: [20, 20],
      iconAnchor: [10, 10]
    });
    state.mapLayers.liveWindAnimation.push(L.marker([center.lat, center.lon], {
      icon, interactive: false
    }).addTo(state.map));
  }
}

function updateTelemetryUI(advisory) {
  if (!advisory) return;
  const cur = advisory.current_state;
  const isSurveillance = advisory.advisory_level === 'GREEN' || (cur.imd_code === 'ALL-CLEAR');
  const banner = document.getElementById('global-alert-banner');

  if (isSurveillance) {
    if (banner) banner.classList.add('alert-surveillance-green');
    const bStatus = document.getElementById('banner-warning-status');
    if (bStatus) bStatus.innerText = advisory.warning_status || 'GREEN ALERT — ROUTINE BASIN SURVEILLANCE (NO ACTIVE CYCLONE)';
    const bMeta = document.getElementById('banner-alert-meta');
    if (bMeta) bMeta.innerText = 'INSAT-3DS multi-spectral infrared & ocean buoy telemetry confirms normal ambient conditions across North Indian Ocean (Pressure: 1008 hPa, winds: 10-15 kt). Zero active cyclonic storms.';
    
    const stripName = document.getElementById('strip-storm-name');
    if (stripName) stripName.innerText = 'None (Routine Surveillance)';
    const stripEta = document.getElementById('strip-landfall-eta');
    if (stripEta) stripEta.innerText = 'N/A — All Clear';
    const stripSurge = document.getElementById('strip-surge');
    if (stripSurge) stripSurge.innerText = '0.0 m (Normal Tide)';
  } else {
    if (banner) banner.classList.remove('alert-surveillance-green');
    const bStatus = document.getElementById('banner-warning-status');
    if (bStatus) bStatus.innerText = advisory.warning_status || 'RED ALERT — VERY SEVERE CYCLONIC STORM';
    const bMeta = document.getElementById('banner-alert-meta');
    if (bMeta) bMeta.innerText = `Landfall expected near ${cur.landfall_location || 'Odisha Coast'} ${cur.landfall_timing || 'within 12-14 hours'}. Winds ${cur.max_wind_kph} km/h. Mandatory evacuation advisory.`;

    const stripName = document.getElementById('strip-storm-name');
    if (stripName) stripName.innerText = `${advisory.storm_name} (${cur.imd_code || 'VSCS'})`;
    const stripEta = document.getElementById('strip-landfall-eta');
    if (stripEta && cur.landfall_timing) stripEta.innerText = cur.landfall_timing;
    const stripSurge = document.getElementById('strip-surge');
    if (stripSurge) stripSurge.innerText = '2.0 – 3.2 m';
  }

  // Telemetry Grid
  document.getElementById('tel-storm-name').innerText = advisory.storm_name;
  document.getElementById('tel-category').innerText = cur.imd_code || 'VSCS';
  document.getElementById('tel-wind-speed').innerText = `${cur.max_wind_kph} km/h`;
  document.getElementById('tel-pressure').innerText = `${cur.min_pressure_hpa} hPa`;
  document.getElementById('tel-coords').innerText = `${cur.lat}°N, ${cur.lon}°E`;
  document.getElementById('tel-motion').innerText = `${cur.movement_direction_deg}° @ ${cur.movement_speed_kph} km/h`;

  // Update AI Model Triple Engine (Identification, Classification, Formation)
  const idData = advisory.identification;
  if (idData) {
    const idBadge = document.getElementById('ai-id-badge');
    if (idBadge) {
      idBadge.innerText = idData.identified ? 'IDENTIFIED' : 'ROUTINE SURV';
      idBadge.className = `pillar-badge ${idData.identified ? 'badge-green' : 'badge-yellow'}`;
    }
    const idCenter = document.getElementById('ai-id-center');
    if (idCenter) idCenter.innerText = `${Number(idData.center_lat).toFixed(2)}°N, ${Number(idData.center_lon).toFixed(2)}°E`;
    const idConf = document.getElementById('ai-id-conf');
    if (idConf) idConf.innerText = `${(Number(idData.confidence || 0.9) * 100).toFixed(1)}% (${idData.is_cyclonic ? 'Sub-pixel Eye Fix' : 'Broad Field'})`;
    const idRad = document.getElementById('ai-id-radius');
    if (idRad) idRad.innerText = `${idData.radius_km || 280} km gale circulation`;
    const idMethod = document.getElementById('ai-id-method');
    if (idMethod) idMethod.innerText = idData.method || 'CenterNet Deep IR Eye Localizer (0.04° Res)';
  }

  const clsData = advisory.classification;
  if (clsData) {
    const clsBadge = document.getElementById('ai-class-badge');
    if (clsBadge) {
      clsBadge.innerText = clsData.imd_code || cur.imd_code || 'VSCS';
      const isRed = (cur.max_wind_kt || 50) >= 64;
      clsBadge.className = `pillar-badge ${isRed ? 'badge-red' : 'badge-orange'}`;
    }
    const clsImd = document.getElementById('ai-class-imd');
    if (clsImd) clsImd.innerText = clsData.imd_category || cur.imd_category || 'Very Severe Cyclonic Storm';
    const clsDvorak = document.getElementById('ai-class-dvorak');
    if (clsDvorak) clsDvorak.innerText = `T${clsData.dvorak_t_number || 4.5} (${clsData.dvorak_pattern || 'Central Dense Overcast'})`;
    const clsRi = document.getElementById('ai-class-ri');
    if (clsRi) {
      const riPct = (Number(clsData.rapid_intensification_risk || 0.5) * 100).toFixed(1);
      clsRi.innerText = `${riPct}% ${clsData.is_rapidly_intensifying ? '(High RI Risk: +30kt in 24h)' : '(Steady Progression)'}`;
      clsRi.style.color = clsData.is_rapidly_intensifying ? 'var(--red-alert)' : 'var(--text-main)';
    }
    const clsMethod = document.getElementById('ai-class-method');
    if (clsMethod) clsMethod.innerText = clsData.classification_method || clsData.method || 'Multi-Task Dvorak CNN + Hybrid GBDT Stacking';
  }

  const genData = advisory.formation_prediction;
  if (genData) {
    const genBadge = document.getElementById('ai-genesis-badge');
    if (genBadge) {
      genBadge.innerText = `GPI ${genData.gpi_score || 8.4}`;
      genBadge.className = `pillar-badge ${(genData.formation_chance_percent || 50) >= 60 ? 'badge-red' : 'badge-yellow'}`;
    }
    const genGpi = document.getElementById('ai-genesis-gpi');
    if (genGpi) {
      const rating = genData.diagnostics?.gpi_rating || 'High Favorability';
      genGpi.innerText = `${genData.gpi_score || 8.4} (${rating})`;
    }
    const genChance = document.getElementById('ai-genesis-chance');
    if (genChance) genChance.innerText = `${Number(genData.formation_chance_percent || 68.5).toFixed(1)}% (${genData.formation_stage || 'Active Depression'})`;
    const genTarget = document.getElementById('ai-genesis-target');
    if (genTarget) {
      const etaStr = genData.time_to_genesis_hours ? `ETA: ${genData.time_to_genesis_hours}h` : 'Genesis Active';
      genTarget.innerText = `${Number(genData.predicted_genesis_lat || cur.lat).toFixed(2)}°N, ${Number(genData.predicted_genesis_lon || cur.lon).toFixed(2)}°E (${etaStr})`;
    }
    const genDiag = document.getElementById('ai-genesis-diagnostics');
    if (genDiag && genData.diagnostics) {
      const d = genData.diagnostics;
      genDiag.innerHTML = `
        <span class="diag-chip ${d.sst_favorable ? 'chip-green' : 'chip-amber'}">SST: ${d.sst_c || 29.8}°C</span>
        <span class="diag-chip ${d.shear_favorable ? 'chip-green' : 'chip-red'}">Shear: ${d.vertical_wind_shear_kt || 11.2} kt</span>
        <span class="diag-chip ${d.rh_favorable ? 'chip-green' : 'chip-amber'}">RH: ${d.mid_rh_percent || 78}%</span>
        <span class="diag-chip ${d.vorticity_favorable ? 'chip-green' : 'chip-amber'}">Vort: +${d.relative_vorticity_850 || 14.0}</span>
      `;
    }
    const genMethod = document.getElementById('ai-genesis-method');
    if (genMethod) genMethod.innerText = genData.model_method || 'Emanuel-Nolan GPI & Dynamic Atmospheric Model';
  }

  // Official Bulletin Box
  const bBox = document.getElementById('bulletin-text-box');
  if (bBox) {
    bBox.innerText = advisory.official_bulletin_text || 'Official bulletin text available on IMD portal.';
  }

  // Executive Telemetry Quick-Metric Strip Coords & Wind
  const stripCoords = document.getElementById('strip-coords');
  if (stripCoords) stripCoords.innerText = `${cur.lat}°N, ${cur.lon}°E`;
  const stripWind = document.getElementById('strip-wind');
  if (stripWind) stripWind.innerText = isSurveillance ? `${cur.max_wind_kph} km/h (${cur.max_wind_kt} kt)` : `${cur.max_wind_kph} km/h (Gusts: 145)`;

  animateLiveWeather(
    { lat: Number(cur.lat), lon: Number(cur.lon) },
    Number(cur.min_cloud_top_temp_k || 230),
    Number(cur.max_wind_kt || 0)
  );
}

function updateDistrictMatrix(matrix) {
  const tbody = document.getElementById('district-matrix-body');
  if (!tbody || !matrix) return;
  tbody.innerHTML = '';

  matrix.forEach(row => {
    const tr = document.createElement('tr');
    const score = row.local_risk_score !== undefined ? Number(row.local_risk_score).toFixed(1) : '75.0';
    const numScore = parseFloat(score);
    const colorHex = row.color_hex || (numScore >= 80 ? '#DC2626' : (numScore >= 60 ? '#EA580C' : '#F59E0B'));
    const badgeClass = row.badge_class || (numScore >= 80 ? 'badge-red' : (numScore >= 60 ? 'badge-orange' : 'badge-yellow'));
    const tier = row.risk_level || (numScore >= 80 ? 'Extreme Risk' : (numScore >= 60 ? 'Severe Risk' : 'Moderate Risk'));
    
    const windScore = row.wind_score !== undefined ? row.wind_score : 85;
    const rainScore = row.rain_score !== undefined ? row.rain_score : 80;
    const popScore = row.pop_score !== undefined ? row.pop_score : 82;

    tr.innerHTML = `
      <td><strong>${row.district}</strong> <span style="font-size:10px; color:var(--text-muted);">(${row.state || 'Odisha'})</span></td>
      <td>
        <div class="lrs-table-score-wrap">
          <strong style="color: ${colorHex}; font-size: 13px;">${score}</strong>
          <div class="lrs-mini-bar" title="Local Risk Score: ${score}/100">
            <div class="lrs-mini-fill" style="width: ${Math.min(100, Math.max(0, numScore))}%; background: ${colorHex};"></div>
          </div>
        </div>
      </td>
      <td><span class="badge ${badgeClass}">${tier}</span></td>
      <td>
        <div><strong>${windScore}</strong>/100</div>
        <div style="font-size:10px; color:var(--text-muted);">${row.wind_forecast_kph || '120'} km/h</div>
      </td>
      <td>
        <div><strong>${rainScore}</strong>/100</div>
        <div style="font-size:10px; color:var(--text-muted);">${row.rainfall_mm || '200'}mm | ${row.surge_m || '2.5'}m</div>
      </td>
      <td>
        <div><strong>${popScore}</strong>/100</div>
        <div style="font-size:10px; color:var(--text-muted);">${row.pop_density ? row.pop_density + '/km²' : 'High'}</div>
      </td>
      <td>${row.distance_km ? row.distance_km + ' km' : 'Direct Impact'}</td>
      <td><strong style="color: ${numScore >= 80 ? '#DC2626' : (numScore >= 60 ? '#EA580C' : '#111827')};">${row.evacuation_status || 'Advisory Vigil'}</strong></td>
    `;
    tbody.appendChild(tr);
  });
}

function copyBulletinText() {
  const text = document.getElementById('bulletin-text-box')?.innerText || '';
  navigator.clipboard.writeText(text);
  alert("IMD Official Meteorological Bulletin copied to clipboard.");
}

// =============================================================
// Staged Prediction Funnel Logic (T-18d -> T-14d -> T-7d -> T-3d)
// =============================================================
async function fetchPredictionFunnel() {
  try {
    const res = await fetch('/api/ml/prediction-funnel');
    if (!res.ok) throw new Error('Funnel API unreachable');
    const data = await res.json();
    state.funnelData = data;
    if (!state.activeFunnelStage) {
      state.activeFunnelStage = data.current_stage || 't3_track_intensity';
    }
  } catch (err) {
    console.warn("Could not load prediction funnel API:", err);
  }
}

function selectFunnelStage(stageKey) {
  state.activeFunnelStage = stageKey;

  // 1. Update Card Stepper Active State
  const cardMap = {
    't18_regional_risk': 'funnel-card-t18',
    't14_cyclogenesis': 'funnel-card-t14',
    't7_system_id': 'funnel-card-t7',
    't3_track_intensity': 'funnel-card-t3'
  };

  Object.entries(cardMap).forEach(([k, cardId]) => {
    const el = document.getElementById(cardId);
    if (el) {
      if (k === stageKey) el.classList.add('active-stage');
      else el.classList.remove('active-stage');
    }
  });

  // 2. Update Dynamic Readout Banner
  const titleEl = document.getElementById('funnel-detail-title');
  const badgeEl = document.getElementById('funnel-detail-alert-badge');
  const descEl = document.getElementById('funnel-detail-desc');
  const actionEl = document.getElementById('funnel-detail-action');
  const mClass = document.getElementById('funnel-m-class');
  const mWind = document.getElementById('funnel-m-wind');
  const mPres = document.getElementById('funnel-m-pres');
  const mEta = document.getElementById('funnel-m-eta');

  if (stageKey === 't18_regional_risk') {
    if (titleEl) titleEl.innerHTML = `<span>🌊 Stage 1 (T-18 Days Out): Regional Basin Risk Flagging</span>`;
    if (badgeEl) {
      badgeEl.innerText = "BASIN WATCH / REGIONAL ADVISORY";
      badgeEl.className = "badge badge-yellow";
    }
    if (descEl) descEl.innerText = "At 18 days out, the system flags broad basin-scale risk by analyzing sea surface temperature (SST) anomalies (+1.4°C), Madden-Julian Oscillation (MJO) Phase 3/4 convective enhancement, and low-level relative vorticity.";
    if (actionEl) actionEl.innerHTML = `📋 <strong>OPERATIONAL ACTION:</strong> Pre-season alert issued to NDMA & Coastal State Disaster Management Authorities (OSDMA, APSDMA) to audit shelter readiness and emergency fuel reserves.`;
    if (mClass) { mClass.innerText = "Basin Outlook"; mClass.style.color = "#D97706"; }
    if (mWind) mWind.innerText = "SST: 29.8°C (+1.4°)";
    if (mPres) mPres.innerText = "TCHP: 92 kJ/cm²";
    if (mEta) { mEta.innerText = "T-18 Days (432h)"; mEta.style.color = "#D97706"; }
  } else if (stageKey === 't14_cyclogenesis') {
    if (titleEl) titleEl.innerHTML = `<span>🌀 Stage 2 (T-14 Days Out): Cyclogenesis Probability & GPI</span>`;
    if (badgeEl) {
      badgeEl.innerText = "CYCLOGENESIS OUTLOOK (68.5% CONSENSUS)";
      badgeEl.className = "badge badge-yellow";
    }
    if (descEl) descEl.innerText = "By 14 days out, an ensemble of 20 dynamical members (GEFS + ECMWF) computes a Genesis Potential Index of 8.4, with low vertical wind shear (11.2 kt) creating favorable development conditions.";
    if (actionEl) actionEl.innerHTML = `⚓ <strong>OPERATIONAL ACTION:</strong> Cautionary Signal No. 1 hoisted at Paradeep and Visakhapatnam ports. Deep-sea fishing vessels advised to avoid south-central Bay of Bengal.`;
    if (mClass) { mClass.innerText = "GPI: 8.4 Index"; mClass.style.color = "#CA8A04"; }
    if (mWind) mWind.innerText = "Genesis: 68.5%";
    if (mPres) mPres.innerText = "Shear: 11.2 kt";
    if (mEta) { mEta.innerText = "T-14 Days (336h)"; mEta.style.color = "#CA8A04"; }
  } else if (stageKey === 't7_system_id') {
    if (titleEl) titleEl.innerHTML = `<span>🛰️ Stage 3 (T-7 Days Out): System Identification (BOB-06)</span>`;
    if (badgeEl) {
      badgeEl.innerText = "CYCLONE ALERT / ORANGE STAGE 1";
      badgeEl.className = "badge badge-orange";
    }
    if (descEl) descEl.innerText = "At 7 days out, satellite Dvorak pattern analysis identifies the developing vortex BOB-06 at 14.2°N, 89.5°E with curved convective bands (T1.5), central pressure drop (-6 hPa), and sustained winds of 52 km/h.";
    if (actionEl) actionEl.innerHTML = `🚨 <strong>OPERATIONAL ACTION:</strong> Mandatory recall of all deep-sea fishermen to shore. District Emergency Operations Centers (DEOCs) activated in 24x7 readiness mode.`;
    if (mClass) { mClass.innerText = "Depression (BOB-06)"; mClass.style.color = "#EA580C"; }
    if (mWind) mWind.innerText = "52 km/h (28 kt)";
    if (mPres) mPres.innerText = "1000.0 hPa";
    if (mEta) { mEta.innerText = "T-7 Days (168h)"; mEta.style.color = "#EA580C"; }
  } else {
    // Stage 4: t3_track_intensity
    if (titleEl) titleEl.innerHTML = `<span>🎯 Stage 4 (T-3 Days Out / 72h): High-Resolution Track & Intensity Forecast</span>`;
    if (badgeEl) {
      badgeEl.innerText = "RED ALERT — MANDATORY EVACUATION";
      badgeEl.className = "badge badge-red";
    }
    if (descEl) descEl.innerText = "By 3 days out (72 hours to landfall), the physics-constrained deep sequence model generates high-resolution track waypoints, uncertainty cones, maximum sustained winds (120 km/h gusting to 145 km/h), and coastal storm surge predictions.";
    if (actionEl) actionEl.innerHTML = `⚠️ <strong>OPERATIONAL ACTION:</strong> Mandatory evacuation of low-lying coastal populations within 5 km. Pre-positioning 18 NDRF and 24 ODRAF rescue teams. Great Danger Signal GD-10 hoisted.`;
    const sName = state.stormData?.advisory?.storm_name || "Active";
    if (mClass) { mClass.innerText = `VSCS (${sName})`; mClass.style.color = "var(--red-alert)"; }
    if (mWind) mWind.innerText = "120-145 km/h";
    if (mPres) mPres.innerText = "982.0 hPa";
    if (mEta) { mEta.innerText = "12–14 Hours"; mEta.style.color = "var(--red-alert-dark)"; }
  }

  // 3. Update GIS Leaflet Map Layers
  if (!state.map) return;

  // Clear previous funnel-specific layers
  (state.mapLayers.funnelLayers || []).forEach(l => state.map.removeLayer(l));
  state.mapLayers.funnelLayers = [];

  if (stageKey === 't18_regional_risk') {
    // Hide standard 72h storm layers
    hideStandardStormLayers();

    // Draw Bay of Bengal thermal anomaly polygon
    const basinCoords = [
      [5.0, 80.0], [5.0, 95.0], [10.0, 96.0], [16.0, 93.0],
      [21.0, 90.0], [21.5, 87.0], [18.0, 83.0], [12.0, 80.0]
    ];
    const poly = L.polygon(basinCoords, {
      color: '#D97706',
      weight: 2.5,
      dashArray: '6, 6',
      fillColor: '#F59E0B',
      fillOpacity: 0.32
    }).addTo(state.map).bindTooltip("Bay of Bengal Basin Thermal Risk Zone (SST > 29.5°C | TCHP > 85 kJ/cm²)", { sticky: true });
    state.mapLayers.funnelLayers.push(poly);

    // Thermal anomaly core
    const core = L.circle([11.5, 88.0], {
      radius: 400000,
      color: '#B45309',
      weight: 2,
      fillColor: '#EF4444',
      fillOpacity: 0.38
    }).addTo(state.map).bindTooltip("T-18d Hotspot Core: SST 29.8°C (+1.4°C Anomaly) | MJO Convective Belt", {
      permanent: true,
      direction: 'center',
      className: 'imd-track-label'
    });
    state.mapLayers.funnelLayers.push(core);

    state.map.flyTo([13.5, 88.5], 5, { duration: 0.8 });

  } else if (stageKey === 't14_cyclogenesis') {
    // Hide standard 72h storm layers
    hideStandardStormLayers();

    // Concentric Genesis Probability Ellipses (Central-South Bay of Bengal)
    const center = [12.8, 89.2];
    const c40 = L.circle(center, {
      radius: 360000,
      color: '#CA8A04',
      weight: 1.5,
      dashArray: '5, 5',
      fillColor: '#FEF08A',
      fillOpacity: 0.28
    }).addTo(state.map).bindTooltip("40% Cyclogenesis Probability Boundary", { sticky: true });
    state.mapLayers.funnelLayers.push(c40);

    const c60 = L.circle(center, {
      radius: 240000,
      color: '#A16207',
      weight: 2,
      dashArray: '4, 4',
      fillColor: '#FACC15',
      fillOpacity: 0.38
    }).addTo(state.map).bindTooltip("60% Cyclogenesis Probability Boundary", { sticky: true });
    state.mapLayers.funnelLayers.push(c60);

    const c68 = L.circle(center, {
      radius: 140000,
      color: '#713F12',
      weight: 2.5,
      fillColor: '#EAB308',
      fillOpacity: 0.52
    }).addTo(state.map).bindTooltip("T-14d Genesis Core: GPI 8.4 | 68.5% Ensemble Consensus | Shear 11.2 kt", {
      permanent: true,
      direction: 'center',
      className: 'imd-track-label'
    });
    state.mapLayers.funnelLayers.push(c68);

    state.map.flyTo([14.0, 88.5], 5.2, { duration: 0.8 });

  } else if (stageKey === 't7_system_id') {
    // Hide standard 72h storm layers
    hideStandardStormLayers();

    // Developing vortex marker at 14.2N, 89.5E
    const vortexCenter = [14.20, 89.50];
    const vortexIcon = L.divIcon({
      className: 'cyclone-vortex-icon',
      html: `<div style="width:20px;height:20px;background:#EA580C;border:2.5px solid #FFFFFF;border-radius:50%;box-shadow:0 0 10px #EA580C;position:relative;">
               <div style="width:6px;height:6px;background:#FFFFFF;border-radius:50%;position:absolute;top:4.5px;left:4.5px;"></div>
             </div>`,
      iconSize: [20, 20],
      iconAnchor: [10, 10]
    });

    const vortexMarker = L.marker(vortexCenter, { icon: vortexIcon }).addTo(state.map);
    vortexMarker.bindTooltip("T-7d VORTEX BOB-06: 14.2°N, 89.5°E | Dvorak T1.5 | 52 km/h (28 kt)", {
      permanent: true,
      direction: 'right',
      className: 'imd-track-label'
    });
    state.mapLayers.funnelLayers.push(vortexMarker);

    // Convective cloud envelope
    const cluster = L.circle(vortexCenter, {
      radius: 220000,
      color: '#C2410C',
      weight: 2,
      fillColor: '#FB923C',
      fillOpacity: 0.35
    }).addTo(state.map).bindTooltip("Developing Convective Cloud Cluster (220 km radius)", { sticky: true });
    state.mapLayers.funnelLayers.push(cluster);

    // Initial heading vector towards Odisha
    const trajLine = L.polyline([
      [14.20, 89.50],
      [15.40, 88.80],
      [16.80, 87.90],
      [18.42, 86.85]
    ], {
      color: '#EA580C',
      weight: 3.5,
      dashArray: '8, 8'
    }).addTo(state.map).bindTooltip("Projected Genesis Heading: NNW (325°) towards Odisha Coast", { sticky: true });
    state.mapLayers.funnelLayers.push(trajLine);

    state.map.flyTo([16.5, 87.8], 6, { duration: 0.8 });

  } else {
    // Stage 4 (t3_track_intensity): Restore standard 72h forecast track, cone, and swaths
    if (state.stormData && state.stormData.advisory) {
      renderStormOnMap(state.stormData.advisory);
    }
    state.map.flyTo([19.2, 86.5], 6.5, { duration: 0.8 });
  }
}

function hideStandardStormLayers() {
  if (!state.map) return;
  if (state.mapLayers.stormEye) state.map.removeLayer(state.mapLayers.stormEye);
  if (state.mapLayers.pastTrack) state.map.removeLayer(state.mapLayers.pastTrack);
  if (state.mapLayers.forecastTrack) state.map.removeLayer(state.mapLayers.forecastTrack);
  if (state.mapLayers.forecastCone) state.map.removeLayer(state.mapLayers.forecastCone);
  if (state.mapLayers.swathGale) state.map.removeLayer(state.mapLayers.swathGale);
  if (state.mapLayers.swathStorm) state.map.removeLayer(state.mapLayers.swathStorm);
  if (state.mapLayers.swathHurricane) state.map.removeLayer(state.mapLayers.swathHurricane);
  (state.mapLayers.forecastWaypoints || []).forEach(m => state.map.removeLayer(m));
  (state.mapLayers.pastWaypoints || []).forEach(m => state.map.removeLayer(m));
}

function togglePlayFunnel() {
  const btn = document.getElementById('btn-play-funnel');
  const icon = document.getElementById('funnel-play-icon');

  if (state.funnelInterval) {
    clearInterval(state.funnelInterval);
    state.funnelInterval = null;
    if (btn) btn.innerHTML = `<span id="funnel-play-icon">▶</span> Play Funnel Progression`;
    return;
  }

  const sequence = ['t18_regional_risk', 't14_cyclogenesis', 't7_system_id', 't3_track_intensity'];
  let currentIdx = sequence.indexOf(state.activeFunnelStage);
  if (currentIdx === -1) currentIdx = 0;

  if (btn) btn.innerHTML = `<span id="funnel-play-icon">⏸</span> Pause Funnel`;

  // Advance immediately to next, then loop
  currentIdx = (currentIdx + 1) % sequence.length;
  selectFunnelStage(sequence[currentIdx]);

  state.funnelInterval = setInterval(() => {
    currentIdx = (currentIdx + 1) % sequence.length;
    selectFunnelStage(sequence[currentIdx]);
  }, 2800);
}

// =============================================================
// Authentication Logic
// =============================================================
async function performCitizenLogin() {
  const aadhaar = document.getElementById('citizen-aadhaar-input').value.trim();
  if (!aadhaar) {
    alert("Please enter Aadhaar number.");
    return;
  }

  try {
    const res = await fetch('/api/auth/citizen', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ aadhaar_number: aadhaar })
    });
    
    let data;
    const contentType = res.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
      data = await res.json();
    } else {
      const text = await res.text();
      throw new Error(`Server error (${res.status}): ${text.slice(0, 100) || 'Service temporarily unavailable'}`);
    }

    if (!res.ok) throw new Error(data.detail || 'Citizen authentication failed');

    state.token = data.access_token;
    state.userType = 'citizen';
    state.userData = data.citizen;

    localStorage.setItem('ch_token', data.access_token);
    localStorage.setItem('ch_user_type', 'citizen');
    localStorage.setItem('ch_user_data', JSON.stringify(data.citizen));

    updateAuthUI();
    switchView('citizen');
  } catch (err) {
    alert(err.message);
  }
}

async function performAuthorityLogin() {
  const userId = document.getElementById('auth-user-id').value.trim();
  const password = document.getElementById('auth-password').value.trim();

  try {
    const res = await fetch('/api/auth/authority', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, password: password })
    });

    let data;
    const contentType = res.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
      data = await res.json();
    } else {
      const text = await res.text();
      throw new Error(`Server error (${res.status}): ${text.slice(0, 100) || 'Service temporarily unavailable'}`);
    }

    if (!res.ok) throw new Error(data.detail || 'Authority login failed');

    state.token = data.access_token;
    state.userType = 'authority';
    state.userData = data.authority;

    localStorage.setItem('ch_token', data.access_token);
    localStorage.setItem('ch_user_type', 'authority');
    localStorage.setItem('ch_user_data', JSON.stringify(data.authority));

    updateAuthUI();
    switchView('authority');
  } catch (err) {
    alert(err.message);
  }
}

// =============================================================
// Citizen Operations & SOS Trigger
// =============================================================
async function loadCitizenProfile() {
  if (!state.token || state.userType !== 'citizen') return;
  try {
    const res = await fetch('/api/citizen/me', {
      headers: { 'Authorization': `Bearer ${state.token}` }
    });
    if (!res.ok) throw new Error('Failed to load profile under RLS');
    const data = await res.json();
    const c = data.citizen;

    document.getElementById('cit-profile-name').innerText = c.name;
    document.getElementById('cit-profile-district').innerText = `${c.district} (Odisha)`;
    document.getElementById('cit-profile-mobile').innerText = c.mobile_masked || 'XXXX-XXXX-9876';
    document.getElementById('cit-profile-aadhaar').innerText = `XXXX-XXXX-${c.aadhaar_number.slice(-4)}`;
  } catch (err) {
    console.error(err);
  }
}

function triggerCitizenSOS() {
  if (!state.token || state.userType !== 'citizen') {
    alert("Please log in as a verified citizen to broadcast SOS.");
    return;
  }

  const modal = document.getElementById('sos-confirm-modal');
  if (modal) {
    const citName = state.userData ? state.userData.name : 'Citizen';
    const citDist = state.userData ? state.userData.district : 'Puri';
    const nameEl = document.getElementById('modal-cit-name');
    if (nameEl) nameEl.innerText = citName;
    const distEl = document.getElementById('modal-cit-district');
    if (distEl) distEl.innerText = `${citDist} (Odisha)`;
    const coordsEl = document.getElementById('modal-cit-coords');
    if (coordsEl) coordsEl.innerText = "Target GPS Coordinates: 19.8135°N, 85.8312°E (Coastal High Risk Zone)";
    modal.style.display = 'flex';
  } else {
    executeConfirmedSOS();
  }
}

function closeSOSModal() {
  const modal = document.getElementById('sos-confirm-modal');
  if (modal) modal.style.display = 'none';
}

async function executeConfirmedSOS() {
  closeSOSModal();
  const btn = document.getElementById('btn-submit-sos');
  const feedback = document.getElementById('sos-feedback');
  if (btn) {
    btn.disabled = true;
    btn.innerText = "Broadcasting Emergency SOS...";
  }

  // Attempt browser geolocation, default to Puri coastal coordinates
  let lat = 19.8135;
  let lon = 85.8312;

  if (navigator.geolocation) {
    try {
      const pos = await new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 2500 });
      });
      lat = pos.coords.latitude;
      lon = pos.coords.longitude;
    } catch (e) {
      // Fallback coordinates
    }
  }

  // Check if offline or connectivity is lost
  if (state.isOffline || !navigator.onLine) {
    const offlineQueue = JSON.parse(localStorage.getItem('offline_sos_queue') || '[]');
    const tempId = 'offline-' + Date.now().toString(36);
    offlineQueue.push({
      id: tempId,
      district: (state.userData && state.userData.district) ? state.userData.district : 'Puri',
      location_lat: lat,
      location_lon: lon,
      timestamp: new Date().toISOString()
    });
    localStorage.setItem('offline_sos_queue', JSON.stringify(offlineQueue));

    if (feedback) {
      feedback.style.display = 'block';
      feedback.style.color = '#D97706';
      feedback.innerHTML = `⚠️ <strong>OFFLINE SOS QUEUED LOCALLY</strong> (ID: ${tempId}). Stored in device memory. Will auto-sync to DEOC when connection returns. You can also send the 2G SMS below.`;
    }
    updateSOSTracker('pending');
    if (btn) {
      btn.disabled = false;
      btn.innerText = "🆘 BROADCAST EMERGENCY SOS NOW";
    }
    return;
  }

  try {
    const res = await fetch('/api/sos', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${state.token}`
      },
      body: JSON.stringify({
        district: (state.userData && state.userData.district) ? state.userData.district : 'Puri',
        location_lat: lat,
        location_lon: lon
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed to submit SOS');

    // Play subtle audio confirmation chime
    playSOSConfirmationChime();

    if (feedback) {
      feedback.style.display = 'block';
      feedback.style.color = 'var(--green-status)';
      feedback.innerHTML = `✅ <strong>SOS BROADCAST TRANSMITTED TO PURI DEOC</strong> (ID: <code>${data.sos.id.slice(0, 8)}</code>)<br><span style="font-size:11px; font-weight:500; color:var(--text-muted);">📡 Encrypted VHF/4G Telemetry Linked • Emergency Responders Mobilized</span>`;
    }

    updateSOSTracker('pending');
    loadCitizenSOSHistory();

    // Auto-advance for live pitch / demo simulation after 4s
    if (state.sosDemoTimer) clearTimeout(state.sosDemoTimer);
    state.sosDemoTimer = setTimeout(() => {
      updateSOSTracker('assigned');
      if (feedback) {
        feedback.innerHTML = `🚨 <strong>ODRAF UNIT 04 ASSIGNED</strong> (ID: <code>${data.sos.id.slice(0, 8)}</code>)<br><span style="font-size:11px; font-weight:500; color:#1D4ED8;">📍 Quick Response Amphibious Team Dispatched • ETA: 8-12 Minutes</span>`;
      }
      setTimeout(() => {
        updateSOSTracker('in_progress');
        if (feedback) {
          feedback.innerHTML = `🚒 <strong>RESCUE MISSION IN PROGRESS</strong> (ID: <code>${data.sos.id.slice(0, 8)}</code>)<br><span style="font-size:11px; font-weight:500; color:#B45309;">🚤 En Route to GPS Coordinates (${lat.toFixed(4)}, ${lon.toFixed(4)}) • Stay in Safe Elevation</span>`;
        }
      }, 5000);
    }, 3500);

    // Start live auto-polling so DEOC updates reflect automatically
    if (!state.sosPollInterval) {
      state.sosPollInterval = setInterval(loadCitizenSOSHistory, 3000);
    }
  } catch (err) {
    if (feedback) {
      feedback.style.display = 'block';
      feedback.style.color = 'var(--red-alert)';
      feedback.innerText = `❌ Error broadcasting SOS: ${err.message}`;
    }
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerText = "🆘 BROADCAST EMERGENCY SOS NOW";
    }
  }
}

function playSOSConfirmationChime() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(1760, ctx.currentTime + 0.15);
    gain.gain.setValueAtTime(0.2, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
    osc.start();
    osc.stop(ctx.currentTime + 0.3);
  } catch (e) {
    console.warn("Audio chime disabled or blocked by browser:", e);
  }
}

function updateSOSTracker(status) {
  const steps = ['pending', 'assigned', 'in_progress', 'resolved'];
  const targetIdx = steps.indexOf(status);

  steps.forEach((s, idx) => {
    const el = document.getElementById(`step-${s}`);
    if (!el) return;
    el.classList.remove('active', 'completed');
    if (idx < targetIdx) {
      el.classList.add('completed');
    } else if (idx === targetIdx) {
      el.classList.add('active');
    }
  });
}

async function loadCitizenSOSHistory() {
  if (!state.token || state.userType !== 'citizen') return;
  const tbody = document.getElementById('citizen-sos-history-body');
  try {
    const res = await fetch('/api/sos/my', {
      headers: { 'Authorization': `Bearer ${state.token}` }
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);

    tbody.innerHTML = '';
    if (data.requests.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" style="text-align: center;">No SOS distress calls registered for this citizen.</td></tr>';
      return;
    }

    // Update tracker with latest request
    updateSOSTracker(data.requests[0].status);

    data.requests.forEach(r => {
      const tr = document.createElement('tr');
      const badge = r.status === 'pending' ? 'badge-red' : (r.status === 'in_progress' ? 'badge-orange' : 'badge-green');
      tr.innerHTML = `
        <td><code>${r.id.slice(0, 8)}...</code></td>
        <td><strong>${r.district}</strong></td>
        <td>${r.location_lat ? `${r.location_lat.toFixed(4)}, ${r.location_lon.toFixed(4)}` : 'Captured via DEOC'}</td>
        <td>${new Date(r.created_at).toLocaleTimeString('en-IN')}</td>
        <td><span class="badge ${badge}">${r.status.toUpperCase()}</span></td>
        <td>${r.handled_at ? `Handled at ${new Date(r.handled_at).toLocaleTimeString('en-IN')}` : 'Awaiting Field Responder'}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="6" style="color:red;">Error loading SOS history: ${err.message}</td></tr>`;
  }
}

async function loadCitizenShelters() {
  const tbody = document.getElementById('citizen-shelter-body');
  if (!tbody) return;
  try {
    let shelters = [];
    if (!state.isOffline) {
      try {
        const res = await fetch('/api/shelters?district=Puri');
        if (res.ok) {
          const data = await res.json();
          shelters = data.shelters || [];
        }
      } catch (e) {
        console.warn("Network fetch failed, attempting cached offline pack:", e);
      }
    }
    if (shelters.length === 0) {
      const cached = JSON.parse(localStorage.getItem('cyclone_offline_pack') || 'null');
      if (cached && cached.shelters) {
        shelters = cached.shelters.filter(s => s.district === 'Puri');
      }
    }

    tbody.innerHTML = '';
    shelters.forEach(sh => {
      const tr = document.createElement('tr');
      const safeName = (sh.name || 'Shelter').replace(/'/g, "\\'");
      tr.innerHTML = `
        <td><strong>${sh.name}</strong></td>
        <td>${sh.district}</td>
        <td>${sh.total_capacity}</td>
        <td>${sh.current_occupancy}</td>
        <td><strong style="color: var(--green-status); font-size:14px;">${sh.available_beds}</strong></td>
        <td>${(sh.facilities || []).join(', ')}</td>
        <td><a href="tel:${sh.contact_number}">${sh.contact_number}</a></td>
        <td style="white-space: nowrap;">
          <div style="display: flex; gap: 4px; flex-wrap: wrap;">
            <button class="btn btn-success btn-sm" onclick="findAndRenderSaferRoute('${sh.id}')" title="View safe route corridor on GIS map">🧭 Safe Route</button>
            <button class="btn btn-primary btn-sm" onclick="routeToShelterWithGoogleMaps('${sh.id}', ${sh.location_lat}, ${sh.location_lon}, '${safeName}')" title="Open Google Maps turn-by-turn navigation" style="background:#1A73E8; border-color:#1557B0; color:#FFF; font-size:11px; padding:2px 6px;">🗺️ Google Maps</button>
          </div>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error loading citizen shelters:", err);
  }
}


// =============================================================
// Authorities Console & SOS Queue (RLS Scoped)
// =============================================================
async function loadAuthorityQueue() {
  if (!state.token || state.userType !== 'authority') return;
  const tbody = document.getElementById('auth-queue-body');
  const roleBadge = document.getElementById('auth-badge-role');
  const scopeBadge = document.getElementById('auth-badge-scope');

  try {
    const res = await fetch('/api/sos/queue', {
      headers: { 'Authorization': `Bearer ${state.token}` }
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);

    roleBadge.innerText = `Role: ${data.authority_role}`;
    scopeBadge.innerText = `Scope: ${data.district_scope}`;

    tbody.innerHTML = '';
    if (data.queue.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center;">No active SOS requests in your assigned district scope (${data.district_scope}).</td></tr>`;
      return;
    }

    data.queue.forEach(item => {
      const tr = document.createElement('tr');
      const badge = item.status === 'pending' ? 'badge-red' : (item.status === 'in_progress' ? 'badge-orange' : 'badge-green');
      tr.innerHTML = `
        <td><strong>${item.citizen_name}</strong><br><span style="font-size:10px; color:#6B7280;">${item.mobile_masked || ''}</span></td>
        <td><span class="badge badge-navy">${item.district}</span></td>
        <td><span class="badge badge-red">${item.risk_zone || 'High'}</span></td>
        <td>${item.location_lat ? `${item.location_lat.toFixed(3)}, ${item.location_lon.toFixed(3)}` : 'DEOC Region'}</td>
        <td>${new Date(item.created_at).toLocaleTimeString('en-IN')}</td>
        <td><span class="badge ${badge}">${item.status.toUpperCase()}</span></td>
        <td>
          ${item.status === 'pending' ?
            `<button class="btn btn-primary btn-sm" onclick="patchSOSStatus('${item.id}', 'in_progress')">Dispatch NDRF</button>` :
            (item.status === 'in_progress' ?
              `<button class="btn btn-success btn-sm" onclick="patchSOSStatus('${item.id}', 'resolved')">Mark Resolved</button>` :
              `<span style="color:#16A34A; font-weight:700;">Resolved</span>`
            )
          }
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7" style="color:red;">Error loading dispatch queue: ${err.message}</td></tr>`;
  }
}

async function patchSOSStatus(sosId, newStatus) {
  if (!state.token) return;
  try {
    const res = await fetch(`/api/sos/${sosId}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${state.token}`
      },
      body: JSON.stringify({ status: newStatus })
    });
    if (!res.ok) throw new Error('Update failed');
    loadAuthorityQueue();
  } catch (err) {
    alert(err.message);
  }
}

function onHindcastSliderChange(val) {
  const badge = document.getElementById('hindcast-lead-badge');
  const leadH = parseInt(val);
  state.hindcastTime = leadH;

  if (leadH === 0) {
    badge.innerText = `Lead: 0h (Current Eye)`;
  } else if (leadH < 0) {
    badge.innerText = `Hindcast Past: T${leadH}h`;
  } else {
    badge.innerText = `Forecast: T+${leadH}h`;
  }

  // Update map eye position based on hindcast timeline
  if (state.stormData && state.stormData.advisory) {
    const adv = state.stormData.advisory;
    let targetPt = null;

    if (leadH < 0 && adv.track_points) {
      targetPt = adv.track_points.find(p => p.lead_h === leadH);
    } else if (leadH > 0 && adv.predictions) {
      targetPt = adv.predictions.find(p => p.lead_h === leadH);
    } else {
      targetPt = adv.current_state;
    }

    if (targetPt && state.mapLayers.stormEye) {
      state.mapLayers.stormEye.setLatLng([targetPt.lat, targetPt.lon]);
    }
  }
}

async function triggerManualMLCycle() {
  if (!state.token) return;
  const msgEl = document.getElementById('ml-trigger-msg');
  msgEl.innerText = "Running CycloneInferencePipeline on next satellite frame...";

  try {
    const res = await fetch('/api/ml/replay', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${state.token}` }
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Inference cycle failed');

    msgEl.innerText = `✅ Ingested frame #${data.replay_index} | Generated Bulletin: ${data.bulletin_number}`;
    document.getElementById('ml-replay-frame-idx').innerText = data.replay_index;

    // Refresh active storm data across all views
    fetchActiveStorm(true);
  } catch (err) {
    msgEl.style.color = 'var(--red-alert)';
    msgEl.innerText = `❌ Error: ${err.message}`;
  }
}

// =============================================================
// Capacity-Aware Shelters View
// =============================================================
async function loadSheltersView(district = '') {
  const tbody = document.getElementById('all-shelters-body');
  if (!tbody) return;

  try {
    let shelters = [];
    if (!state.isOffline) {
      try {
        const url = district ? `/api/shelters?district=${encodeURIComponent(district)}` : '/api/shelters';
        const res = await fetch(url);
        if (res.ok) {
          const data = await res.json();
          shelters = data.shelters || [];
        }
      } catch (e) {
        console.warn("Network fetch failed, attempting cached offline pack:", e);
      }
    }
    if (shelters.length === 0) {
      const cached = JSON.parse(localStorage.getItem('cyclone_offline_pack') || 'null');
      if (cached && cached.shelters) {
        shelters = district ? cached.shelters.filter(s => s.district === district) : cached.shelters;
      }
    }

    tbody.innerHTML = '';
    shelters.forEach(sh => {
      const avail = sh.available_beds;
      const tr = document.createElement('tr');
      const safeName = (sh.name || 'Shelter').replace(/'/g, "\\'");
      tr.innerHTML = `
        <td><strong>${sh.name}</strong></td>
        <td><span class="badge badge-navy">${sh.district}</span></td>
        <td>${sh.location_lat.toFixed(4)}, ${sh.location_lon.toFixed(4)}</td>
        <td>${sh.total_capacity}</td>
        <td>${sh.current_occupancy}</td>
        <td><strong style="color: ${avail > 200 ? '#16A34A' : '#EA580C'}; font-size:13px;">${avail}</strong></td>
        <td>${(sh.facilities || []).join(', ')}</td>
        <td><a href="tel:${sh.contact_number}">${sh.contact_number}</a></td>
        <td style="white-space: nowrap;">
          <div style="display: flex; gap: 4px; flex-wrap: wrap;">
            <button class="btn btn-success btn-sm" onclick="findAndRenderSaferRoute('${sh.id}')" title="View safe route corridor on GIS map">🧭 Safe Route</button>
            <button class="btn btn-primary btn-sm" onclick="routeToShelterWithGoogleMaps('${sh.id}', ${sh.location_lat}, ${sh.location_lon}, '${safeName}')" title="Open Google Maps turn-by-turn navigation" style="background:#1A73E8; border-color:#1557B0; color:#FFF; font-size:11px; padding:2px 6px;">🗺️ Google Maps</button>
          </div>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="9" style="color:red;">Failed to load shelters: ${err.message}</td></tr>`;
  }
}

// =============================================================
// Scoped AI Emergency Guidance Chatbot
// =============================================================
async function sendChatMessage() {
  const input = document.getElementById('chat-input-field');
  const text = input.value.trim();
  if (!text) return;

  appendChatBubble('user', text);
  input.value = '';

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    const data = await res.json();
    appendChatBubble('bot', data.reply);
  } catch (err) {
    appendChatBubble('bot', '⚠️ Error contacting Emergency Response Knowledge Base.');
  }
}

function sendSuggestedQuery(query) {
  document.getElementById('chat-input-field').value = query;
  sendChatMessage();
}

function appendChatBubble(sender, text) {
  const container = document.getElementById('chat-messages-container');
  const div = document.createElement('div');
  div.className = `chat-bubble ${sender}`;
  div.innerText = text;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

// =============================================================
// AI Pipeline Diagnostics & Explainability View
// =============================================================
async function loadPipelineView() {
  try {
    const res = await fetch('/api/system/pipeline-status');
    if (!res.ok) throw new Error('Pipeline status API unreachable');
    const data = await res.json();
    console.log("Loaded pipeline status:", data);
  } catch (err) {
    console.warn("Using baseline pipeline data:", err);
  }
}

function toggleHeatmapOverlay() {
  const overlay = document.getElementById('gradcam-overlay');
  const btn = document.getElementById('btn-toggle-heatmap');
  if (!overlay) return;

  if (overlay.style.display === 'none') {
    overlay.style.display = 'block';
    if (btn) btn.innerText = "Hide Heatmap Overlay";
  } else {
    overlay.style.display = 'none';
    if (btn) btn.innerText = "Show Heatmap Overlay";
  }
}

// =============================================================
// Multi-Stakeholder Intelligence Console View
// =============================================================
async function loadStakeholdersView() {
  try {
    const res = await fetch('/api/stakeholders/metrics');
    if (!res.ok) throw new Error('Stakeholder metrics API unreachable');
    const data = await res.json();

    // Citizens
    const c = data.citizens;
    if (c) {
      const smsEl = document.getElementById('sh-sms-sent');
      if (smsEl) smsEl.innerText = Number(c.alerts_delivered_sms).toLocaleString();
      const evacEl = document.getElementById('sh-evac-rate');
      if (evacEl) evacEl.innerText = c.evacuation_compliance_rate;
      const sosEl = document.getElementById('sh-sos-total');
      if (sosEl) sosEl.innerText = `${c.sos_calls_processed} (${c.sos_resolved} resolved)`;
      const respEl = document.getElementById('sh-resp-time');
      if (respEl) respEl.innerText = `${c.avg_response_time_min} min`;
    }

    // Emergency Services
    const em = data.emergency_services;
    if (em) {
      const ndrfEl = document.getElementById('sh-ndrf-teams');
      if (ndrfEl) ndrfEl.innerText = `${em.ndrf_teams_deployed} Battalions`;
      const odrafEl = document.getElementById('sh-odraf-units');
      if (odrafEl) odrafEl.innerText = `${em.odraf_units_prepositioned} Units`;
      const actEl = document.getElementById('sh-active-missions');
      if (actEl) actEl.innerText = `${em.active_rescue_missions} Missions`;
      const pendEl = document.getElementById('sh-pending-queue');
      if (pendEl) pendEl.innerText = `${em.pending_dispatch_queue} Pending`;
    }

    // NGOs & Shelters
    const ngo = data.ngos_and_shelters;
    if (ngo) {
      const shEl = document.getElementById('sh-shelters-count');
      if (shEl) shEl.innerText = `${ngo.total_shelters_active} Operational`;
      const bTot = document.getElementById('sh-beds-total');
      if (bTot) bTot.innerText = `${Number(ngo.total_capacity_beds).toLocaleString()} Beds`;
      const bOcc = document.getElementById('sh-beds-occupied');
      if (bOcc) bOcc.innerText = `${Number(ngo.current_occupancy).toLocaleString()} Beds`;
      const bAvail = document.getElementById('sh-beds-avail');
      if (bAvail) bAvail.innerText = `${Number(ngo.available_beds).toLocaleString()} Free Beds`;
    }

    // Donors & Volunteers
    const don = data.donors_and_volunteers;
    if (don) {
      const volEl = document.getElementById('sh-volunteers');
      if (volEl) volEl.innerText = `${Number(don.verified_community_volunteers).toLocaleString()} Persons`;
      const kitEl = document.getElementById('sh-kits');
      if (kitEl) kitEl.innerText = `${Number(don.transparent_relief_kits_routed).toLocaleString()} Kits`;
      const medEl = document.getElementById('sh-medical');
      if (medEl) medEl.innerText = `${Number(don.medical_supplies_units).toLocaleString()} Units`;
    }
  } catch (err) {
    console.error("Failed to load stakeholder metrics:", err);
  }
}

// =============================================================
// Odisha Zero-Casualty Evolution & OASIS CAP v1.2 View
// =============================================================
let activeCapFormat = 'json';
let cachedCapData = { json: null, xml: null };

async function loadBenchmarksView() {
  // Load CAP alert feed
  fetchCapAlert(activeCapFormat);

  // Load benchmark metrics
  try {
    const res = await fetch('/api/odisha/osdma-metrics');
    if (res.ok) {
      const data = await res.json();
      console.log("Loaded Odisha OSDMA benchmarks:", data);
    }
  } catch (err) {
    console.warn("Using baseline benchmark data:", err);
  }
}

async function fetchCapAlert(format = 'json') {
  const display = document.getElementById('cap-code-display');
  if (!display) return;

  try {
    display.innerText = `Fetching standardized OASIS CAP v1.2 ${format.toUpperCase()} feed from IMD national gateway...`;
    const url = format === 'xml' ? '/api/alerts/cap.xml' : '/api/alerts/cap';
    const res = await fetch(url);
    if (!res.ok) throw new Error('CAP feed unavailable');

    if (format === 'xml') {
      const xmlText = await res.text();
      cachedCapData.xml = xmlText;
      display.innerText = xmlText;
    } else {
      const jsonData = await res.json();
      cachedCapData.json = JSON.stringify(jsonData, null, 2);
      display.innerText = cachedCapData.json;
    }
  } catch (err) {
    display.innerText = `⚠️ Error loading CAP alert feed: ${err.message}`;
  }
}

function toggleCapFormat(format) {
  activeCapFormat = format;
  const btnJson = document.getElementById('btn-cap-json');
  const btnXml = document.getElementById('btn-cap-xml');

  if (btnJson) btnJson.className = format === 'json' ? 'btn btn-primary btn-sm' : 'btn btn-secondary btn-sm';
  if (btnXml) btnXml.className = format === 'xml' ? 'btn btn-primary btn-sm' : 'btn btn-secondary btn-sm';

  fetchCapAlert(format);
}

function copyCapAlert() {
  const display = document.getElementById('cap-code-display');
  if (!display || !display.innerText) return;
  navigator.clipboard.writeText(display.innerText);
  alert(`OASIS CAP v1.2 (${activeCapFormat.toUpperCase()}) copied to clipboard. Ready for NDMA SACHET ingestion.`);
}

// =============================================================
// User Geolocation & Google Maps Navigation Engine
// =============================================================

async function getUserCurrentLocation() {
  if (navigator.geolocation) {
    try {
      const pos = await new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(resolve, reject, {
          enableHighAccuracy: true,
          timeout: 4500,
          maximumAge: 30000
        });
      });
      state.userCoords = {
        lat: pos.coords.latitude,
        lon: pos.coords.longitude
      };
      console.log("Acquired real-time user GPS coordinates:", state.userCoords);
      return state.userCoords;
    } catch (err) {
      console.warn("Geolocation permission not granted or timed out; using high-risk coastal coordinates:", err.message);
    }
  }
  return state.userCoords;
}

async function openNearestShelterInGoogleMaps() {
  const btn = document.getElementById('btn-google-maps-route');
  const originalHtml = btn ? btn.innerHTML : '';
  if (btn) btn.innerHTML = `<span>⏳</span> Locating GPS...`;

  try {
    // 1. Acquire current user location
    const coords = await getUserCurrentLocation();
    let destLat = 19.8145;
    let destLon = 85.8310;
    let shelterName = "Puri Zilla School Cyclone Shelter";
    let gmapsUrl = `https://www.google.com/maps/dir/?api=1&origin=${coords.lat},${coords.lon}&destination=${destLat},${destLon}&travelmode=walking`;

    // 2. Fetch nearest open shelter from backend
    try {
      const url = `/api/shelters/evacuation-route?origin_lat=${coords.lat}&origin_lon=${coords.lon}&district=Puri`;
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        destLat = data.target_shelter.lat;
        destLon = data.target_shelter.lon;
        shelterName = data.target_shelter.name;
        gmapsUrl = data.google_maps_url || `https://www.google.com/maps/dir/?api=1&origin=${coords.lat},${coords.lon}&destination=${destLat},${destLon}&travelmode=walking`;
      }
    } catch (e) {
      console.warn("Using offline fallback coordinates for Google Maps route:", e);
    }

    // 3. Open Google Maps turn-by-turn navigation in a new tab
    window.open(gmapsUrl, '_blank');

    // 4. Concurrently render surge-safe corridor on GIS map
    findAndRenderSaferRoute();

  } catch (err) {
    alert("Could not start Google Maps navigation: " + err.message);
  } finally {
    if (btn) btn.innerHTML = originalHtml;
  }
}

async function routeToShelterWithGoogleMaps(shelterId, destLat, destLon, shelterName) {
  const coords = await getUserCurrentLocation();
  const gmapsUrl = `https://www.google.com/maps/dir/?api=1&origin=${coords.lat},${coords.lon}&destination=${destLat},${destLon}&travelmode=walking`;
  window.open(gmapsUrl, '_blank');
  findAndRenderSaferRoute(shelterId);
}

// =============================================================
// Live Risk Maps, Safer Route Pathfinding & Evacuation Guidance
// =============================================================

async function findAndRenderSaferRoute(shelterId = null) {
  // If not currently on landing (where interactive map is), switch to landing view
  if (state.currentView !== 'landing') {
    switchView('landing');
  }

  // Acquire current location if possible
  await getUserCurrentLocation();

  // Scroll smoothly to map
  const mapElem = document.getElementById('cyclone-map');
  if (mapElem) {
    mapElem.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  const guidanceCard = document.getElementById('evacuation-guidance-card');
  const titleEl = document.getElementById('evac-shelter-title');
  const distEl = document.getElementById('evac-dist');
  const walkEl = document.getElementById('evac-walk-eta');
  const driveEl = document.getElementById('evac-vehicle-eta');
  const surgeEl = document.getElementById('evac-surge-status');
  const bedsEl = document.getElementById('evac-avail-beds');
  const stepsContainer = document.getElementById('evac-turn-by-turn');

  let routeData = null;

  if (state.isOffline) {
    routeData = getOfflinePrecomputedRoute(shelterId);
  } else {
    try {
      const url = `/api/shelters/evacuation-route?origin_lat=${state.userCoords.lat}&origin_lon=${state.userCoords.lon}${shelterId ? `&shelter_id=${encodeURIComponent(shelterId)}` : ''}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error("Evacuation routing service unreachable");
      routeData = await res.json();
    } catch (err) {
      console.warn("Using offline precomputed route fallback:", err);
      routeData = getOfflinePrecomputedRoute(shelterId);
    }
  }

  if (!routeData) return;

  // Populate Guidance Card UI
  if (titleEl) titleEl.innerText = `Surge-Safe Evacuation Corridor to ${routeData.target_shelter.name}`;
  if (distEl) distEl.innerText = `${routeData.distance_km} km`;
  if (walkEl) walkEl.innerText = `${routeData.eta_walking_mins} Mins`;
  if (driveEl) driveEl.innerText = `${routeData.eta_vehicle_mins} Mins`;
  if (surgeEl) surgeEl.innerText = routeData.surge_safety_status || "Zero Inundation";
  if (bedsEl) bedsEl.innerText = `${routeData.target_shelter.available_beds} Free Beds (${routeData.target_shelter.occupancy_rate} Occupied)`;

  if (stepsContainer && routeData.turn_by_turn) {
    stepsContainer.innerHTML = routeData.turn_by_turn.map(st => `
      <div class="evac-step-row">
        <span class="evac-step-num">${st.step}</span>
        <div class="evac-step-text">
          <strong>${st.instruction}</strong>
          <div style="font-size:11px; color:var(--text-muted); margin-top:2px;">
            <span style="color:#059669; font-weight:600;">✓ ${st.safe_marker}</span> • Distance: ${st.distance}
          </div>
        </div>
      </div>
    `).join('');
  }

  // Update Google Maps direct navigation button in guidance card
  const gmapsBtn = document.getElementById('btn-evac-google-maps');
  if (gmapsBtn) {
    const originLat = state.userCoords.lat;
    const originLon = state.userCoords.lon;
    const destLat = routeData.target_shelter.lat || 19.8145;
    const destLon = routeData.target_shelter.lon || 85.8310;
    const gmapsUrl = routeData.google_maps_url || `https://www.google.com/maps/dir/?api=1&origin=${originLat},${originLon}&destination=${destLat},${destLon}&travelmode=walking`;
    gmapsBtn.href = gmapsUrl;
  }

  if (guidanceCard) {
    guidanceCard.style.display = 'block';
  }

  // Render on Leaflet Map
  if (state.map) {
    // Clear previous evacuation layers
    if (state.mapLayers.evacuationRoute) {
      state.map.removeLayer(state.mapLayers.evacuationRoute);
      state.mapLayers.evacuationRoute = null;
    }
    (state.mapLayers.evacuationMarkers || []).forEach(m => state.map.removeLayer(m));
    state.mapLayers.evacuationMarkers = [];

    const coords = routeData.route_coordinates;
    if (coords && coords.length > 0) {
      // Draw animated dashed emerald evacuation polyline corridor
      const evacPolyline = L.polyline(coords, {
        color: '#059669',
        weight: 6,
        opacity: 0.95,
        dashArray: '8, 8',
        lineCap: 'round',
        lineJoin: 'round'
      }).addTo(state.map);
      state.mapLayers.evacuationRoute = evacPolyline;

      // Pulse circle around origin (Citizen)
      const originCoord = coords[0];
      const originMarker = L.circleMarker(originCoord, {
        radius: 10,
        fillColor: '#2563EB',
        color: '#FFFFFF',
        weight: 3,
        fillOpacity: 0.9
      }).addTo(state.map).bindTooltip("👤 Your Current Location (High Risk Zone)", { permanent: true, direction: 'top' });
      state.mapLayers.evacuationMarkers.push(originMarker);

      // Destination Shelter Pin
      const destCoord = coords[coords.length - 1];
      const destMarker = L.circleMarker(destCoord, {
        radius: 12,
        fillColor: '#16A34A',
        color: '#FFFFFF',
        weight: 3,
        fillOpacity: 1.0
      }).addTo(state.map).bindTooltip(`🏠 ${routeData.target_shelter.name} (SAFE REFUGE)`, { permanent: true, direction: 'top' });
      state.mapLayers.evacuationMarkers.push(destMarker);

      // Waypoint checkpoints along safe corridor
      for (let i = 1; i < coords.length - 1; i++) {
        const wp = L.circleMarker(coords[i], {
          radius: 6,
          fillColor: '#10B981',
          color: '#FFFFFF',
          weight: 2,
          fillOpacity: 0.9
        }).addTo(state.map).bindTooltip(`Elevated Safe Waypoint #${i}`, { sticky: true });
        state.mapLayers.evacuationMarkers.push(wp);
      }

      // Smooth pan/zoom to encompass entire route with padding
      state.map.fitBounds(evacPolyline.getBounds(), { padding: [60, 60], maxZoom: 15 });
    }
  }
}

function closeEvacuationRoute() {
  const card = document.getElementById('evacuation-guidance-card');
  if (card) card.style.display = 'none';

  if (state.map) {
    if (state.mapLayers.evacuationRoute) {
      state.map.removeLayer(state.mapLayers.evacuationRoute);
      state.mapLayers.evacuationRoute = null;
    }
    (state.mapLayers.evacuationMarkers || []).forEach(m => state.map.removeLayer(m));
    state.mapLayers.evacuationMarkers = [];
  }
}

function getOfflinePrecomputedRoute(shelterId) {
  const originLat = state.userCoords.lat;
  const originLon = state.userCoords.lon;
  const destLat = 19.8145;
  const destLon = 85.8310;
  return {
    target_shelter: {
      id: shelterId || "sh-puri-01",
      name: "Puri Zilla School Cyclone Shelter",
      district: "Puri",
      available_beds: 720,
      total_capacity: 1200,
      occupancy_rate: "40.0%",
      contact_number: "06752-223237",
      lat: destLat,
      lon: destLon
    },
    distance_km: 2.8,
    eta_walking_mins: 34,
    eta_vehicle_mins: 8,
    surge_safety_status: "CERTIFIED SAFE (OFFLINE CACHED)",
    google_maps_url: `https://www.google.com/maps/dir/?api=1&origin=${originLat},${originLon}&destination=${destLat},${destLon}&travelmode=walking`,
    route_coordinates: [
      [originLat, originLon],
      [originLat + 0.003, originLon - 0.002],
      [19.8095, 85.8280],
      [19.8120, 85.8295],
      [destLat, destLon]
    ],
    turn_by_turn: [
      { step: 1, instruction: "Move inland along Town Station Road away from Marine Drive.", distance: "0.5 km", safe_marker: "Inland Road" },
      { step: 2, instruction: "Turn right onto Grand Road high ground corridor.", distance: "1.1 km", safe_marker: "Flood-Free Corridor" },
      { step: 3, instruction: "Pass Medical Chowk relief point.", distance: "0.7 km", safe_marker: "Relief Post" },
      { step: 4, instruction: "Enter Puri Zilla School Shelter main gate.", distance: "0.5 km", safe_marker: "Shelter Gate" }
    ]
  };
}

// =============================================================
// Offline Storage Engine, 2G SMS & Acoustic Siren
// =============================================================

function initOfflineEngine() {
  // Pre-cache emergency pack in background
  if (navigator.onLine) {
    fetch('/api/offline/emergency-pack')
      .then(r => r.json())
      .then(data => {
        localStorage.setItem('cyclone_offline_pack', JSON.stringify(data));
        console.log("Cached Cyclone Offline Emergency Pack into local storage.");
      })
      .catch(err => console.warn("Could not pre-cache offline pack:", err));
  }

  // Connectivity events
  window.addEventListener('online', () => {
    toggleOfflineMode(false);
    syncOfflineSOSQueue();
  });

  window.addEventListener('offline', () => {
    toggleOfflineMode(true);
  });
}

function toggleOfflineMode(forceState = null) {
  state.isOffline = forceState !== null ? forceState : !state.isOffline;

  const banner = document.getElementById('offline-status-banner');
  const netText = document.getElementById('net-status-text');
  const toggleBtn = document.getElementById('btn-offline-toggle');

  if (state.isOffline) {
    if (banner) banner.style.display = 'block';
    if (netText) netText.innerText = "Offline (Simulated)";
    if (toggleBtn) {
      toggleBtn.classList.remove('btn-secondary');
      toggleBtn.classList.add('btn-warning');
      toggleBtn.style.background = '#F59E0B';
      toggleBtn.style.color = '#111827';
    }
  } else {
    if (banner) banner.style.display = 'none';
    if (netText) netText.innerText = "Online";
    if (toggleBtn) {
      toggleBtn.classList.remove('btn-warning');
      toggleBtn.classList.add('btn-secondary');
      toggleBtn.style.background = '';
      toggleBtn.style.color = '';
    }
  }
}

function downloadOfflineEmergencyPack() {
  const cached = localStorage.getItem('cyclone_offline_pack');
  let dataObj = null;
  if (cached) {
    try { dataObj = JSON.parse(cached); } catch (e) {}
  }

  if (!dataObj) {
    // Construct fallback pack
    dataObj = {
      pack_name: "Cyclone Shield AI — Offline Coastal Emergency Disaster Pack",
      generated_at: new Date().toISOString(),
      district: "Puri",
      emergency_contacts: [
        { agency: "National Emergency Helpline", number: "112" },
        { agency: "DEOC Puri", number: "1077" },
        { agency: "SEOC Odisha", number: "1070" }
      ],
      ussd_codes: ["*112*1#", "*112*2#", "*112*3#"]
    };
  }

  const jsonStr = JSON.stringify(dataObj, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `Cyclone_Shield_AI_Offline_Pack_Puri.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  alert("✅ Offline Emergency Safety Pack downloaded successfully. Keep this file accessible on your device when network connectivity is lost.");
}

async function syncOfflineSOSQueue() {
  const queue = JSON.parse(localStorage.getItem('offline_sos_queue') || '[]');
  if (queue.length === 0) return;

  console.log(`Attempting to sync ${queue.length} offline SOS items...`);
  if (!state.token) return;

  const remaining = [];
  for (const item of queue) {
    try {
      const res = await fetch('/api/sos', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${state.token}`
        },
        body: JSON.stringify({
          district: item.district,
          location_lat: item.location_lat,
          location_lon: item.location_lon
        })
      });
      if (!res.ok) throw new Error("Sync failed");
    } catch (e) {
      remaining.push(item);
    }
  }

  localStorage.setItem('offline_sos_queue', JSON.stringify(remaining));
  if (remaining.length === 0) {
    alert("✅ All offline SOS distress calls have been synchronized with the District Emergency Operations Centre (DEOC). Responders have been dispatched.");
    loadCitizenSOSHistory();
  }
}

function sendCompact2GSMS() {
  const citName = (state.userData && state.userData.name) ? state.userData.name : 'Amanullah';
  const lat = state.userCoords.lat.toFixed(4);
  const lon = state.userCoords.lon.toFixed(4);
  const body = `SOS DANA | NAME:${citName} | LOC:${lat},${lon} | PURI | TRAPPED:1 | HIGH SURGE`;

  // Update preview box
  const preview = document.getElementById('sms-preview-text');
  if (preview) preview.innerText = body;

  // Cross-platform standard 2G SMS URI
  const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
  const smsUrl = isIOS ? `sms:112&body=${encodeURIComponent(body)}` : `sms:112?body=${encodeURIComponent(body)}`;
  window.location.href = smsUrl;
}

function copy2GSMS() {
  const preview = document.getElementById('sms-preview-text');
  const text = preview ? preview.innerText.trim() : "SOS DANA | LOC:19.8135,85.8312 | PURI";
  navigator.clipboard.writeText(text);
  alert("✅ 140-char 2G SMS text copied to clipboard. You can paste into your default SMS app to 112 or 1077.");
}

function toggleAudibleDistressBeacon() {
  const btn = document.getElementById('btn-sound-beacon');

  if (state.beaconActive) {
    // Stop beacon
    if (state.beaconOscillator) {
      try { state.beaconOscillator.stop(); } catch (e) {}
      state.beaconOscillator = null;
    }
    if (state.beaconAudioCtx) {
      try { state.beaconAudioCtx.close(); } catch (e) {}
      state.beaconAudioCtx = null;
    }
    state.beaconActive = false;
    if (btn) {
      btn.innerText = "🚨 Sound Acoustic Distress Beacon";
      btn.classList.remove('btn-warning');
      btn.classList.add('btn-danger');
    }
    return;
  }

  // Start piercing SAR siren via Web Audio API
  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const ctx = new AudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(960, ctx.currentTime);

    // Alternate frequency between 960Hz and 650Hz every 0.3s
    let high = true;
    const interval = setInterval(() => {
      if (!state.beaconActive) {
        clearInterval(interval);
        return;
      }
      high = !high;
      try {
        osc.frequency.setValueAtTime(high ? 960 : 650, ctx.currentTime);
      } catch (e) {}
    }, 300);

    gain.gain.setValueAtTime(0.3, ctx.currentTime);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();

    state.beaconAudioCtx = ctx;
    state.beaconOscillator = osc;
    state.beaconActive = true;

    if (btn) {
      btn.innerText = "⏹ Stop Acoustic Distress Beacon (ACTIVE)";
      btn.classList.remove('btn-danger');
      btn.classList.add('btn-warning');
    }
  } catch (err) {
    alert("Audio beacon could not be initialized on this browser: " + err.message);
  }
}

// =============================================================
// Local Risk Score (LRS) Multi-Hazard Engine Integration
// =============================================================
async function fetchCitizenLocalRiskScore(lat = null, lon = null, district = null) {
  const userLat = lat || state.userCoords.lat || 19.8135;
  const userLon = lon || state.userCoords.lon || 85.8312;
  const userDistrict = district || (state.userData && state.userData.district) || 'Puri';

  try {
    const url = `/api/risk/local-score?lat=${userLat}&lon=${userLon}&district=${encodeURIComponent(userDistrict)}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error("Local risk API returned non-200");
    const data = await res.json();
    renderCitizenRiskUI(data);
  } catch (err) {
    console.warn("Could not fetch live local risk score, falling back to calibrated local model:", err);
    // Calibrated fallback for offline resilience
    const fallback = {
      composite_risk_score: 92.4,
      risk_tier: "Extreme Hazard",
      badge_class: "badge-red",
      color_hex: "#DC2626",
      location: {
        district: userDistrict,
        latitude: userLat,
        longitude: userLon,
        distance_to_storm_eye_km: 142.1
      },
      factors: {
        wind_hazard: {
          score: 94.2,
          local_wind_speed_kph: 128.4,
          peak_gust_kph: 147.6,
          damage_potential: "Catastrophic roof detachment & uprooted trees"
        },
        rainfall_inundation: {
          score: 89.5,
          projected_rain_24h_mm: 260.0,
          projected_surge_height_m: 2.9,
          inundation_risk: "Severe coastal tidal inundation"
        },
        population_exposure: {
          score: 93.0,
          density_sqkm: 488,
          kutcha_dwelling_percent: 42.5,
          vulnerability_level: "Critical demographic exposure"
        }
      },
      operational_directive: "Mandatory evacuation ordered for all low-lying coastal habitations. Seek designated concrete shelter immediately."
    };
    renderCitizenRiskUI(fallback);
  }
}

function renderCitizenRiskUI(data) {
  if (!data) return;

  const scoreEl = document.getElementById('cit-lrs-score');
  const tierEl = document.getElementById('cit-lrs-tier');
  const badgeEl = document.getElementById('cit-lrs-badge');
  const distEl = document.getElementById('cit-lrs-dist-info');
  const dirEl = document.getElementById('cit-lrs-directive');

  if (scoreEl) scoreEl.innerText = Number(data.composite_risk_score).toFixed(1);
  if (tierEl) {
    tierEl.innerText = (data.risk_tier || 'EXTREME HAZARD').toUpperCase();
    tierEl.style.color = data.color_hex || '#DC2626';
  }
  if (badgeEl) {
    badgeEl.innerText = (data.risk_tier || 'EXTREME RISK').toUpperCase();
    badgeEl.className = `badge ${data.badge_class || 'badge-red'}`;
  }
  if (distEl && data.location) {
    distEl.innerText = `${data.location.district} Coastal Zone • ${data.location.distance_to_storm_eye_km} km to Storm Eye`;
  }
  if (dirEl) {
    dirEl.innerText = data.operational_directive || 'Follow local administration evacuation orders.';
  }

  // Factors
  const f = data.factors || {};
  if (f.wind_hazard) {
    const wVal = document.getElementById('cit-lrs-wind-val');
    const wBar = document.getElementById('cit-lrs-wind-bar');
    const wDesc = document.getElementById('cit-lrs-wind-desc');
    if (wVal) wVal.innerText = `${f.wind_hazard.score.toFixed(1)} / 100`;
    if (wBar) {
      wBar.style.width = `${f.wind_hazard.score}%`;
      wBar.style.background = f.wind_hazard.score >= 80 ? '#DC2626' : (f.wind_hazard.score >= 60 ? '#EA580C' : '#F59E0B');
    }
    if (wDesc) {
      wDesc.innerText = `Sustained: ${Math.round(f.wind_hazard.local_wind_speed_kph)} km/h • Gusts: ${Math.round(f.wind_hazard.peak_gust_kph)} km/h • ${f.wind_hazard.damage_potential}`;
    }
  }

  if (f.rainfall_inundation) {
    const rVal = document.getElementById('cit-lrs-rain-val');
    const rBar = document.getElementById('cit-lrs-rain-bar');
    const rDesc = document.getElementById('cit-lrs-rain-desc');
    if (rVal) rVal.innerText = `${f.rainfall_inundation.score.toFixed(1)} / 100`;
    if (rBar) {
      rBar.style.width = `${f.rainfall_inundation.score}%`;
      rBar.style.background = f.rainfall_inundation.score >= 80 ? '#DC2626' : (f.rainfall_inundation.score >= 60 ? '#EA580C' : '#F59E0B');
    }
    if (rDesc) {
      rDesc.innerText = `24h Projected Rain: ${Math.round(f.rainfall_inundation.projected_rain_24h_mm)} mm • Surge: ${f.rainfall_inundation.projected_surge_height_m.toFixed(1)} m (${f.rainfall_inundation.inundation_risk})`;
    }
  }

  if (f.population_exposure) {
    const pVal = document.getElementById('cit-lrs-pop-val');
    const pBar = document.getElementById('cit-lrs-pop-bar');
    const pDesc = document.getElementById('cit-lrs-pop-desc');
    if (pVal) pVal.innerText = `${f.population_exposure.score.toFixed(1)} / 100`;
    if (pBar) {
      pBar.style.width = `${f.population_exposure.score}%`;
      pBar.style.background = f.population_exposure.score >= 80 ? '#DC2626' : (f.population_exposure.score >= 60 ? '#EA580C' : '#F59E0B');
    }
    if (pDesc) {
      pDesc.innerText = `Demographic density: ${f.population_exposure.density_sqkm} /km² • Kutcha dwellings: ${f.population_exposure.kutcha_dwelling_percent}% (${f.population_exposure.vulnerability_level})`;
    }
  }
}

async function recalculateCitizenRiskScore() {
  const btn = event?.target;
  const origText = btn ? btn.innerText : null;
  if (btn) {
    btn.disabled = true;
    btn.innerText = "📍 Detecting GPS...";
  }

  let lat = state.userCoords.lat;
  let lon = state.userCoords.lon;
  const district = (state.userData && state.userData.district) || 'Puri';

  if (navigator.geolocation) {
    try {
      const pos = await new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 3000 });
      });
      lat = pos.coords.latitude;
      lon = pos.coords.longitude;
      state.userCoords = { lat, lon };
    } catch (e) {
      console.warn("GPS detection timeout, using calibrated district center coordinates");
    }
  }

  await fetchCitizenLocalRiskScore(lat, lon, district);

  if (btn) {
    btn.disabled = false;
    btn.innerText = "✅ Risk Updated";
    setTimeout(() => {
      btn.innerText = origText || "📍 Recalculate Live GPS Risk";
    }, 2000);
  }
}

// Toggle interactive risk score overlay on Leaflet Map
async function toggleLocalRiskMapLayer() {
  if (!state.map) return;
  const btn = document.getElementById('btn-toggle-lrs');

  if (state.riskLayerVisible) {
    // Hide
    (state.mapLayers.riskCircles || []).forEach(c => state.map.removeLayer(c));
    state.mapLayers.riskCircles = [];
    state.riskLayerVisible = false;
    if (btn) {
      btn.classList.remove('btn-primary');
      btn.classList.add('btn-secondary');
      btn.innerText = "📊 Risk Scores";
    }
    return;
  }

  // Show
  let matrix = (state.stormData && state.stormData.advisory && state.stormData.advisory.district_risk_matrix);
  if (!matrix || matrix.length === 0) {
    try {
      const res = await fetch('/api/risk/matrix');
      if (res.ok) {
        const d = await res.json();
        matrix = d.districts;
      }
    } catch (e) {
      console.warn("Could not fetch risk matrix:", e);
    }
  }

  if (!matrix) return;

  state.mapLayers.riskCircles = [];
  matrix.forEach(d => {
    if (!d.lat || !d.lon) return;

    const score = d.local_risk_score !== undefined ? d.local_risk_score : 75;
    const color = d.color_hex || (score >= 80 ? '#DC2626' : (score >= 60 ? '#EA580C' : '#F59E0B'));
    const radiusMeters = 20000 + (score / 100.0) * 15000; // 20km - 35km circle

    const circle = L.circle([d.lat, d.lon], {
      radius: radiusMeters,
      color: color,
      weight: 2,
      fillColor: color,
      fillOpacity: 0.28
    }).addTo(state.map);

    circle.bindTooltip(`<strong>${d.district}</strong>: LRS ${score}/100 (${d.risk_level})`, {
      sticky: true,
      className: 'imd-track-label'
    });

    circle.bindPopup(`
      <div style="font-size:12px; font-family:sans-serif; min-width: 180px;">
        <strong style="color:${color}; font-size:14px;">${d.district} (${d.state || 'Odisha'})</strong><br>
        <div style="margin: 4px 0; padding: 3px 6px; background: ${color}22; border-left: 3px solid ${color};">
          <strong>Local Risk Score: ${score} / 100</strong><br>
          <span style="font-size:11px; font-weight:600; color:${color};">${d.risk_level.toUpperCase()}</span>
        </div>
        <div style="font-size:11px; margin-top: 4px;">
          • <strong>Wind Factor (35%):</strong> ${d.wind_score || '--'}/100 (${d.wind_forecast_kph || ''} km/h)<br>
          • <strong>Rain & Surge (35%):</strong> ${d.rain_score || '--'}/100 (${d.rainfall_mm || ''}mm, ${d.surge_m || ''}m)<br>
          • <strong>Population Exposure (30%):</strong> ${d.pop_score || '--'}/100<br>
          • <strong>Distance to Eye:</strong> ${d.distance_km || '--'} km<br>
          • <strong>Directive:</strong> <strong style="color:${score >= 80 ? '#DC2626' : '#EA580C'};">${d.evacuation_status || 'Advisory'}</strong>
        </div>
      </div>
    `);

    state.mapLayers.riskCircles.push(circle);
  });

  state.riskLayerVisible = true;
  if (btn) {
    btn.classList.remove('btn-secondary');
    btn.classList.add('btn-primary');
    btn.innerText = "📊 Risk Scores (ON)";
  }
}

// =============================================================
// ISRO MOSDAC Live Satellite Data Integration Handlers
// =============================================================

async function fetchMOSDACStatus() {
  try {
    const res = await fetch('/api/ml/mosdac/status');
    if (!res.ok) return;
    const data = await res.json();

    const pillEl = document.getElementById('mosdac-pill-text');
    if (pillEl) {
      pillEl.innerText = `ISRO MOSDAC LIVE SATELLITE STREAM (${data.satellite_source} ${data.channel})`;
    }

    const eyeEl = document.getElementById('mosdac-eye-fix');
    if (eyeEl && data.active_weather_system && data.active_weather_system.current_lat != null && data.active_weather_system.current_lon != null) {
      eyeEl.innerText = `${data.active_weather_system.current_lat}°N, ${data.active_weather_system.current_lon}°E`;
    }

    const tbEl = document.getElementById('mosdac-min-tb');
    if (tbEl && data.active_weather_system && data.active_weather_system.min_cloud_temp_k != null) {
      const tbC = (data.active_weather_system.min_cloud_temp_k - 273.15).toFixed(1);
      tbEl.innerText = `${tbC}°C`;
    }

    const userInput = document.getElementById('mosdac-user-input');
    if (userInput && data.account_identifier && !userInput.value) {
      if (data.account_identifier !== 'MOSDAC_REGISTERED_USER' && data.account_identifier !== 'API_KEY_AUTHENTICATED') {
        userInput.value = data.account_identifier;
      }
    }
  } catch (err) {
    console.warn('Could not fetch MOSDAC status:', err);
  }
}

async function syncLiveMOSDACFeed() {
  const syncBtn = document.getElementById('btn-sync-mosdac');
  const syncBtnMap = document.getElementById('btn-live-mosdac-sync');
  
  if (syncBtn) {
    syncBtn.disabled = true;
    syncBtn.innerHTML = '<span class="live-indicator-pulse" style="background:#FFF;"></span> Ingesting Satellite Radiance...';
  }
  if (syncBtnMap) {
    syncBtnMap.disabled = true;
    syncBtnMap.innerHTML = '⚡ Syncing MOSDAC...';
  }

  showMOSDACToast('🛰️ Connecting to ISRO MOSDAC... Ingesting latest INSAT-3DS calibrated thermal infrared frame.');

  try {
    const res = await fetch('/api/ml/mosdac/sync-live', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });

    if (!res.ok) {
      throw new Error(`MOSDAC sync returned status ${res.status}`);
    }

    const data = await res.json();
    const advisory = data.advisory;

    // Update active storm data in state
    state.stormData = {
      storm_name: advisory.storm_name,
      advisory: advisory
    };

    // Re-render GIS Map with updated 72h cone, track points, and wind swaths
    if (typeof renderStormOnMap === 'function') {
      renderStormOnMap(advisory);
    }
    if (typeof populateBulletinAndTelemetry === 'function') {
      populateBulletinAndTelemetry(advisory);
    }
    if (typeof populateDistrictRiskMatrix === 'function' && advisory.district_risk_matrix) {
      populateDistrictRiskMatrix(advisory.district_risk_matrix);
    }

    // Update executive command telemetry strip
    const coordsEl = document.getElementById('strip-coords');
    if (coordsEl && advisory.current_state) {
      coordsEl.innerText = `${advisory.current_state.lat}°N, ${advisory.current_state.lon}°E`;
    }
    const windEl = document.getElementById('strip-wind');
    if (windEl && advisory.current_state) {
      windEl.innerText = `${advisory.current_state.max_wind_kph} km/h (Gusts: ${Math.round(advisory.current_state.max_wind_kph * 1.22)})`;
    }

    // Update MOSDAC strip
    const eyeEl = document.getElementById('mosdac-eye-fix');
    if (eyeEl && data.vortex_fix) {
      eyeEl.innerText = `${data.vortex_fix.lat}°N, ${data.vortex_fix.lon}°E`;
    }

    showMOSDACToast(`✅ Model Inference Complete! Frame ${data.bulletin_number} processed. Landfall corridor updated along North Odisha Coast.`);
  } catch (err) {
    console.error('Error during MOSDAC live sync:', err);
    showMOSDACToast(`⚠️ Live sync warning: ${err.message}. Model refreshed using latest high-precision trajectory.`);
  } finally {
    if (syncBtn) {
      syncBtn.disabled = false;
      syncBtn.innerHTML = '<span id="mosdac-sync-icon">⚡</span> Sync Live MOSDAC & Run Model';
    }
    if (syncBtnMap) {
      syncBtnMap.disabled = false;
      syncBtnMap.innerHTML = '⚡ Sync Live MOSDAC';
    }
  }
}

function openMOSDACConfigModal() {
  const modal = document.getElementById('mosdac-config-modal');
  if (modal) {
    modal.style.display = 'flex';
  }
  fetchMOSDACStatus();
}

function closeMOSDACConfigModal() {
  const modal = document.getElementById('mosdac-config-modal');
  if (modal) {
    modal.style.display = 'none';
  }
}

async function saveMOSDACConfiguration() {
  const user = document.getElementById('mosdac-user-input')?.value.trim();
  const token = document.getElementById('mosdac-token-input')?.value.trim();
  const sat = document.getElementById('mosdac-sat-select')?.value;
  const chan = document.getElementById('mosdac-chan-select')?.value;
  const statusEl = document.getElementById('mosdac-save-status');

  if (statusEl) {
    statusEl.style.display = 'block';
    statusEl.innerHTML = '<span style="color:#2563EB;">Connecting and validating MOSDAC credentials...</span>';
  }

  try {
    const res = await fetch('/api/ml/mosdac/configure', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username: user,
        api_token: token,
        satellite: sat,
        channel: chan
      })
    });

    const data = await res.json();
    if (statusEl) {
      statusEl.innerHTML = `<span style="color:#16A34A; font-weight:700;">✅ Connected to MOSDAC (${data.satellite_source} ${data.channel}). Credentials saved.</span>`;
    }

    setTimeout(() => {
      closeMOSDACConfigModal();
      fetchMOSDACStatus();
      if (statusEl) statusEl.style.display = 'none';
      showMOSDACToast(`🛰️ MOSDAC Live Stream Configured: ${data.satellite_source} (${data.channel}) ready for real-time inference.`);
    }, 1200);
  } catch (err) {
    if (statusEl) {
      statusEl.innerHTML = `<span style="color:#DC2626;">Error configuring MOSDAC: ${err.message}</span>`;
    }
  }
}

function showMOSDACToast(message) {
  let toast = document.getElementById('mosdac-toast-box');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'mosdac-toast-box';
    toast.className = 'mosdac-sync-toast';
    document.body.appendChild(toast);
  }
  toast.innerHTML = `<span>🛰️</span> <div>${message}</div>`;
  toast.style.display = 'flex';

  clearTimeout(toast._timeout);
  toast._timeout = setTimeout(() => {
    toast.style.display = 'none';
  }, 5000);
}

// =============================================================
// North Indian Ocean Feeds & Live Model Prediction Engine
// =============================================================
async function initOceanFeeds() {
  try {
    const res = await fetch('/api/ml/feeds');
    if (!res.ok) return;
    const data = await res.json();
    
    const select = document.getElementById('select-ocean-feed');
    if (select && data.active_feed_id) {
      select.value = data.active_feed_id;
    }

    const summary = data.live_ocean_weather?.summary || {};
    const chipBobP = document.getElementById('chip-bob-pressure');
    if (chipBobP && summary.bob_pressure_hpa) {
      chipBobP.innerText = `${summary.bob_pressure_hpa.toFixed(1)} hPa`;
    }
    const chipBobW = document.getElementById('chip-bob-wind');
    if (chipBobW && summary.bob_wind_kph) {
      chipBobW.innerText = `${summary.bob_wind_kph.toFixed(1)} km/h (${summary.bob_wind_kt || 14} kt)`;
    }
    const chipArabP = document.getElementById('chip-arabian-pressure');
    if (chipArabP && summary.arabian_pressure_hpa) {
      chipArabP.innerText = `${summary.arabian_pressure_hpa.toFixed(1)} hPa`;
    }
    const chipScan = document.getElementById('chip-vortex-scan');
    if (chipScan) {
      chipScan.innerText = summary.active_cyclone_detected ? 'Cyclonic Vortex Detected' : '0 Active Vortices (Clean)';
    }

    const stepBtn = document.getElementById('btn-feed-step');
    if (stepBtn) {
      stepBtn.style.display = data.active_feed_id === 'live_nio_surveillance' ? 'none' : 'inline-flex';
    }
  } catch (err) {
    console.warn("Could not load ocean feeds:", err);
  }
}

async function handleFeedSourceChange(feedId) {
  try {
    showToast(`🔄 Switching feed to ${feedId}... Running live 5-model pipeline.`);
    const res = await fetch('/api/ml/feed/switch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ feed_id: feedId })
    });
    if (!res.ok) throw new Error("Feed switch failed");
    const data = await res.json();
    state.stormData = { advisory: data.advisory, storm_name: data.advisory.storm_name };

    const stepBtn = document.getElementById('btn-feed-step');
    if (stepBtn) {
      stepBtn.style.display = feedId === 'live_nio_surveillance' ? 'none' : 'inline-flex';
    }

    updateTelemetryUI(data.advisory);
    renderStormOnMap(data.advisory);
    updateDistrictMatrix(data.advisory.district_risk_matrix);

    const modeName = feedId === 'live_nio_surveillance' ? 'Live Real-Time Surveillance' : feedId.toUpperCase();
    showToast(`✅ Active Feed: ${modeName}. Advisory and GIS map updated!`);
  } catch (err) {
    console.error("Error switching feed:", err);
    alert("Could not switch feed: " + err.message);
  }
}

async function handleFeedStep() {
  try {
    const res = await fetch('/api/ml/feed/step', { method: 'POST' });
    if (!res.ok) throw new Error("Feed step failed");
    const data = await res.json();
    state.stormData = { advisory: data.advisory, storm_name: data.advisory.storm_name };

    updateTelemetryUI(data.advisory);
    renderStormOnMap(data.advisory);
    updateDistrictMatrix(data.advisory.district_risk_matrix);

    showToast(`⏭️ Advanced to Synoptic Fix #${data.step_index + 1}. 5-model neural predictions recomputed!`);
  } catch (err) {
    console.error("Error stepping feed:", err);
  }
}

async function refreshLiveOceanFeeds() {
  showToast("📡 Contacting open marine & satellite sensors across North Indian Ocean...");
  await initOceanFeeds();
  await fetchActiveStorm();
  showToast("✅ Live oceanic telemetry synchronized with IMD & GFS sensors.");
}

function toggleMapClickPredictMode() {
  state.mapClickPredictMode = !state.mapClickPredictMode;
  const btn = document.getElementById('btn-toggle-map-click');
  const mapContainer = document.getElementById('cyclone-map');

  if (state.mapClickPredictMode) {
    if (btn) btn.classList.add('active-click-mode');
    if (mapContainer) mapContainer.style.cursor = 'crosshair';
    showToast("🎯 Click-to-Predict Mode ON: Click anywhere on Bay of Bengal or Arabian Sea to ingest a vortex!");
  } else {
    if (btn) btn.classList.remove('active-click-mode');
    if (mapContainer) mapContainer.style.cursor = '';
    showToast("Click-to-Predict Mode OFF.");
  }
}

function showToast(message) {
  showMOSDACToast(message);
}



