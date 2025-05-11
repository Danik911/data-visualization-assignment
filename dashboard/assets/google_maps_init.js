// Google Maps initialization helper

// Ensure dash_clientside object exists
window.dash_clientside = window.dash_clientside || {};

// VERIFY GOOGLE MAPS API IS AVAILABLE (can be a local helper or global if needed elsewhere)
function checkGoogleMapsApiInternal() {
  if (typeof google === 'undefined' || typeof google.maps === 'undefined') {
    console.error('Google Maps API not loaded');
    const mapContainer = document.getElementById('google-price-map-container'); // Assumes this ID
    if (mapContainer) {
      mapContainer.innerHTML = `
        <div class="alert alert-danger">
          <strong>Error:</strong> Google Maps API failed to load. 
          Please check your API key and internet connection.
        </div>
      `;
    }
    return false;
  }
  console.log('Google Maps API is loaded');
  return true;
}

// LOAD MARKERCLUSTERER LIBRARY (local helper)
function loadMarkerClustererInternal(callback) {
  if (window.markerClustererLoaded) { // Use a global flag for the library itself
    callback();
    return;
  }
  try {
    if (typeof MarkerClusterer !== 'undefined') { // Check for the global MarkerClusterer object
      window.markerClustererLoaded = true;
      callback();
      return;
    }
    const script = document.createElement('script');
    script.src = "https://unpkg.com/@googlemaps/markerclusterer@2.0.8/dist/index.min.js";
    script.async = true;
    script.onload = function() {
      console.log('MarkerClusterer loaded successfully');
      window.markerClustererLoaded = true;
      callback();
    };
    script.onerror = function() {
      console.error('Failed to load MarkerClusterer');
      callback(); // Continue without clustering
    };
    document.head.appendChild(script);
  } catch (error) {
    console.error('Error loading MarkerClusterer:', error);
    callback(); // Continue without clustering
  }
}

// INITIALIZE MAP WITH MARKERS (local helper)
function initializeMapWithMarkersInternal(data, mapContainerId) {
  console.log('RENDER_DEBUG: initializeMapWithMarkersInternal CALLED. Container ID:', mapContainerId);
  const mapContainer = document.getElementById(mapContainerId);
  if (!mapContainer) {
      console.error('RENDER_DEBUG: Map container with ID ' + mapContainerId + ' not found.');
      return;
  }
  try {
    if (!checkGoogleMapsApiInternal()) {
      console.error('RENDER_DEBUG: checkGoogleMapsApiInternal failed.');
      return;
    }
    console.log('RENDER_DEBUG: checkGoogleMapsApiInternal PASSED.');
    
    const mapData = typeof data === 'string' ? JSON.parse(data) : data;
    console.log("RENDER_DEBUG: Map Data received and parsed by initializeMapWithMarkersInternal:", mapData);
    
    if (mapData.error) {
      console.error('Map data error:', mapData.error);
      mapContainer.innerHTML = `<div class="alert alert-warning">Error: ${mapData.error}</div>`;
      return;
    }
    if (!mapData || !mapData.center || !mapData.data) {
      console.error('Invalid map data structure for initializeMapWithMarkersInternal:', mapData);
      mapContainer.innerHTML = `<div class="alert alert-warning">Error: Invalid map data format</div>`;
      return;
    }
    
    const center = mapData.center || { lat: 41.6, lng: -93.6 };
    if (typeof center.lat !== 'number' || typeof center.lng !== 'number' || isNaN(center.lat) || isNaN(center.lng)) {
      console.error('Invalid map center:', center);
      center.lat = 41.6; center.lng = -93.6;
    }
    
    // Manage map instance, markers, and infoWindow globally or pass them around if preferred
    if (window.googleMapInstance && window.googleMapInstance.getDiv() && window.googleMapInstance.getDiv().id === mapContainerId) {
        console.log(`Re-using existing map instance for container ${mapContainerId}.`);
        if (window.mapMarkers && window.mapMarkers.length > 0) {
            for (let marker of window.mapMarkers) { marker.setMap(null); }
        }
        window.mapMarkers = [];
        if (window.markerClustererInstance) { window.markerClustererInstance.clearMarkers(); }
    } else {
        console.log(`Creating new Google Map instance for ${mapContainerId} at: ${center.lat}, ${center.lng}`);
        window.googleMapInstance = new google.maps.Map(mapContainer, {
            center: center, zoom: mapData.zoom || 12, mapTypeId: google.maps.MapTypeId.ROADMAP,
            mapTypeControl: true, streetViewControl: true, fullscreenControl: true
        });
        window.infoWindow = new google.maps.InfoWindow();
        window.mapMarkers = [];
    }

    if (mapData.data && Array.isArray(mapData.data) && mapData.data.length > 0) {
      let minPrice, maxPrice;
      if (mapData.stats && typeof mapData.stats.price_min !== 'undefined' && typeof mapData.stats.price_max !== 'undefined') {
        minPrice = mapData.stats.price_min; maxPrice = mapData.stats.price_max;
      } else {
        const prices = mapData.data.filter(p => typeof p.Sale_Price === 'number' && !isNaN(p.Sale_Price)).map(p => p.Sale_Price);
        if (prices.length === 0) { console.error('No valid price data for range calculation'); return; }
        minPrice = Math.min(...prices); maxPrice = Math.max(...prices);
      }
      const priceRange = maxPrice - minPrice || 1;
      console.log(`Adding ${mapData.data.length} markers. Price range: $${minPrice}-$${maxPrice}`);

      for (let property of mapData.data) {
        const lat = parseFloat(property.Latitude);
        const lng = parseFloat(property.Longitude);
        const price = parseFloat(property.Sale_Price);

        if (isNaN(lat) || isNaN(lng) || isNaN(price) || !isFinite(lat) || !isFinite(lng) || !isFinite(price) || lat < -90 || lat > 90 || lng < -180 || lng > 180) {
          console.warn('Skipping invalid property data:', { lat: property.Latitude, lng: property.Longitude, price: property.Sale_Price });
          continue;
        }
        const normalizedPrice = (price - minPrice) / priceRange;
        const hue = (1 - normalizedPrice) * 120;
        const color = `hsl(${hue}, 100%, 50%)`;
        const marker = new google.maps.Marker({
          position: { lat: lat, lng: lng },
          icon: { path: google.maps.SymbolPath.CIRCLE, fillColor: color, fillOpacity: 0.8, strokeColor: 'white', strokeWeight: 0.5, scale: 7 },
          title: `Price: $${price.toLocaleString()}\nType: ${property.Bldg_Type_Display || property.Bldg_Type}`
        });
        marker.addListener('click', () => {
          window.infoWindow.setContent(`
            <div style="font-family: Arial, sans-serif; font-size: 14px;">
              <strong>Price:</strong> $${price.toLocaleString()}<br>
              <strong>Type:</strong> ${property.Bldg_Type_Display || property.Bldg_Type}<br>
              <strong>Area:</strong> ${property.Lot_Area} sqft<br>
              <strong>Built:</strong> ${property.Year_Built}
            </div>
          `);
          window.infoWindow.open(window.googleMapInstance, marker);
        });
        window.mapMarkers.push(marker);
      }
      console.log(`Added ${window.mapMarkers.length} valid markers.`);
      
      if (typeof MarkerClusterer === 'function' && window.markerClustererLoaded) {
        console.log('Using MarkerClusterer');
        if (window.markerClustererInstance) { window.markerClustererInstance.clearMarkers(); }
        window.markerClustererInstance = new MarkerClusterer({ map: window.googleMapInstance, markers: window.mapMarkers });
      } else {
        console.log('MarkerClusterer not available/loaded. Adding markers directly.');
        for(let m of window.mapMarkers) { m.setMap(window.googleMapInstance); }
      }
    } else {
      console.log('No data points to add as markers.');
    }
  } catch (error) {
    console.error('Error in initializeMapWithMarkersInternal:', error);
    if (mapContainer) {
        mapContainer.innerHTML = '<div class="alert alert-danger">An error occurred while displaying the map.</div>';
    }
  }
}

// Assign to the dash_clientside namespace
window.dash_clientside.googleMaps = {
    initMap: function(data, mapContainerIdFromCallback /* Optional, but good for explicitness */) {
        const mapContainerId = mapContainerIdFromCallback || 'google-price-map-container'; // Default if not provided
        console.log('RENDER_DEBUG: dash_clientside.googleMaps.initMap CALLED. Data type:', typeof data, 'Container ID:', mapContainerId);
        
        if (typeof data === 'string') {
            console.log('RENDER_DEBUG: Raw string data in dash_clientside.googleMaps.initMap (first 500 chars):', data.substring(0, 500));
        }

        // Wait for container to be ready
        const checkContainerInterval = setInterval(function() {
            const containerElement = document.getElementById(mapContainerId);
            if (containerElement) {
                clearInterval(checkContainerInterval);
                loadMarkerClustererInternal(function() { // Use internal helper
                    initializeMapWithMarkersInternal(data, mapContainerId); // Use internal helper
                });
            } else {
                console.log(`RENDER_DEBUG: Waiting for map container ${mapContainerId} in dash_clientside.googleMaps.initMap...`);
            }
        }, 100);
        return ''; // Clientside functions should return a value for an Output
    }
};

console.log('Google Maps initialization script (google_maps_init.js) parsed and dash_clientside.googleMaps namespace defined.');

// The DOMContentLoaded listener can be very minimal or removed if not strictly needed for other setup.
window.addEventListener('DOMContentLoaded', function() {
  console.log('Google Maps init script DOMContentLoaded event fired. Map initialization is now handled via dash_clientside.googleMaps.initMap.');
});

// Cleanup of old MutationObserver if it exists from previous versions of this file
// This is speculative, assuming 'observer' might have been a global var.
if (window.observer && typeof window.observer.disconnect === 'function') {
    console.log("Disconnecting old MutationObserver from google_maps_init.js.");
    window.observer.disconnect();
    window.observer = null; 
}

// Ensure other global variables are initialized if they are expected to be.
// window.googleMapInstance, window.mapMarkers, window.infoWindow, 
// window.markerClustererInstance, window.markerClustererLoaded are managed within the functions.
// Setting markerClustererLoaded to false initially.
window.markerClustererLoaded = window.markerClustererLoaded || false;

/*
  The old MutationObserver logic has been removed as the clientside_callback approach is now primary.
  If you had specific reasons for the MutationObserver (e.g., reacting to direct DOM manipulations 
  of the data div from other JS sources), you might need to re-evaluate.
  For Dash dcc.Store updates, the clientside_callback is the standard way.
*/