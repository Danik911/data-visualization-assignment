// Google Maps initialization helper

// VERIFY GOOGLE MAPS API IS AVAILABLE (defined globally)
function checkGoogleMapsApi() {
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

// LOAD MARKERCLUSTERS LIBRARY (defined globally)
function loadMarkerClusterer(callback) {
  if (window.markerClustererLoaded) {
    callback();
    return;
  }
  try {
    if (typeof markerClusterer !== 'undefined') {
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

// INITIALIZE MAP WITH MARKERS (defined globally)
// This function now takes mapContainerId as an argument
function initializeMapWithMarkers(data, mapContainerId) {
  console.log('RENDER_DEBUG: initializeMapWithMarkers CALLED. Container ID:', mapContainerId);
  const mapContainer = document.getElementById(mapContainerId);
  if (!mapContainer) {
      console.error('RENDER_DEBUG: Map container with ID ' + mapContainerId + ' not found in initializeMapWithMarkers.');
      return;
  }
  try {
    if (!checkGoogleMapsApi()) {
      console.error('RENDER_DEBUG: checkGoogleMapsApi failed in initializeMapWithMarkers');
      return;
    }
    console.log('RENDER_DEBUG: checkGoogleMapsApi PASSED.');
    
    const mapData = typeof data === 'string' ? JSON.parse(data) : data;
    console.log("RENDER_DEBUG: Map Data received and parsed:", mapData);
    
    if (mapData.error) {
      console.error('Map data error:', mapData.error);
      mapContainer.innerHTML = `<div class="alert alert-warning">Error: ${mapData.error}</div>`;
      return;
    }
    if (!mapData || !mapData.center || !mapData.data) {
      console.error('Invalid map data structure:', mapData);
      mapContainer.innerHTML = `<div class="alert alert-warning">Error: Invalid map data format</div>`;
      return;
    }
    
    const center = mapData.center || { lat: 41.6, lng: -93.6 };
    console.log('RENDER_DEBUG: Calculated center:', center);
    if (typeof center.lat !== 'number' || typeof center.lng !== 'number' || isNaN(center.lat) || isNaN(center.lng)) {
      console.error('Invalid map center:', center);
      center.lat = 41.6; center.lng = -93.6;
    }
    
    if (window.googleMapInstance && window.googleMapInstance.getDiv() && window.googleMapInstance.getDiv().id === mapContainerId) {
        console.log(`Re-using existing map instance for container ${mapContainerId}.`);
        // Clear existing markers and clusterer before adding new ones
        if (window.mapMarkers && window.mapMarkers.length > 0) {
            for (let marker of window.mapMarkers) {
                marker.setMap(null);
            }
        }
        window.mapMarkers = [];
        if (window.markerClustererInstance) {
            window.markerClustererInstance.clearMarkers();
        }
    } else {
        console.log(`Creating new Google Map instance for container ${mapContainerId} centered at: ${center.lat}, ${center.lng}`);
        window.googleMapInstance = new google.maps.Map(mapContainer, {
            center: center,
            zoom: mapData.zoom || 12,
            mapTypeId: google.maps.MapTypeId.ROADMAP,
            mapTypeControl: true, streetViewControl: true, fullscreenControl: true
        });
        window.infoWindow = new google.maps.InfoWindow(); // Create one info window to be reused
        window.mapMarkers = [];
    }

    // ... (rest of the marker creation logic from your original file, ensure it uses window.googleMapInstance)
    // Add markers for properties
    if (mapData.data && Array.isArray(mapData.data) && mapData.data.length > 0) {
      let minPrice, maxPrice;
      if (mapData.stats && typeof mapData.stats.price_min !== 'undefined' && typeof mapData.stats.price_max !== 'undefined') {
        minPrice = mapData.stats.price_min;
        maxPrice = mapData.stats.price_max;
      } else {
        const prices = mapData.data.filter(item => typeof item.Sale_Price === 'number' && !isNaN(item.Sale_Price)).map(item => item.Sale_Price);
        if (prices.length === 0) { console.error('No valid price data for range calculation'); return; }
        minPrice = Math.min(...prices);
        maxPrice = Math.max(...prices);
      }
      const priceRange = maxPrice - minPrice || 1;
      console.log(`Adding ${mapData.data.length} markers. Price range: $${minPrice}-$${maxPrice}`);

      for (let property of mapData.data) {
        if (property === null || typeof property !== 'object') { invalidMarkers++; continue; }
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
          // map: window.googleMapInstance, // Add to clusterer instead
          icon: {
            path: google.maps.SymbolPath.CIRCLE,
            fillColor: color,
            fillOpacity: 0.8,
            strokeColor: 'white',
            strokeWeight: 0.5,
            scale: 7
          },
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
      
      // Use MarkerClusterer if available
      if (typeof MarkerClusterer === 'function' && window.markerClustererLoaded) {
        console.log('Using MarkerClusterer');
        if (window.markerClustererInstance) {
            window.markerClustererInstance.clearMarkers(); // Clear old markers if any
        }
        window.markerClustererInstance = new MarkerClusterer({ map: window.googleMapInstance, markers: window.mapMarkers });
      } else {
        console.log('MarkerClusterer not available or not loaded. Adding markers directly to map.');
        for(let m of window.mapMarkers) { m.setMap(window.googleMapInstance); }
      }
    } else {
      console.log('No data points to add as markers.');
    }

  } catch (error) {
    console.error('Error in initializeMapWithMarkers:', error);
    if (mapContainer) {
        mapContainer.innerHTML = '<div class="alert alert-danger">An error occurred while displaying the map.</div>';
    }
  }
}

// MAIN FUNCTION TO INITIALIZE MAP (defined globally)
// This function is called by the Dash clientside_callback
// It now takes mapContainerId as an argument
window.initGoogleMap = function(data, mapContainerId) {
  console.log('RENDER_DEBUG: initGoogleMap CALLED. Data type:', typeof data, 'Container ID:', mapContainerId);
  if (typeof data === 'string') {
    console.log('RENDER_DEBUG: Raw string data in initGoogleMap (first 500 chars):', data.substring(0, 500));
  }
  
  // Wait for container to be ready (important if initGoogleMap is called before DOM is fully ready for the container)
  const checkContainerInterval = setInterval(function() {
    const containerElement = document.getElementById(mapContainerId);
    if (containerElement) {
      clearInterval(checkContainerInterval);
      // Load MarkerClusterer and then initialize the map
      loadMarkerClusterer(function() {
        initializeMapWithMarkers(data, mapContainerId);
      });
    } else {
        console.log(`RENDER_DEBUG: Waiting for map container ${mapContainerId}...`);
    }
  }, 100); // Check every 100ms
};

// DOMContentLoaded listener (simplified)
window.addEventListener('DOMContentLoaded', function() {
  console.log('Google Maps initialization script DOMContentLoaded event fired.');
  // Any setup that MUST run after DOM is ready but is not part of initGoogleMap can go here.
  // For now, initGoogleMap is called by the Dash clientside callback.
});

// Global variable for map instance and markers (optional, but can be useful for debugging)
// window.googleMapInstance = null; // Now initialized inside initializeMapWithMarkers
// window.mapMarkers = []; // Now initialized inside initializeMapWithMarkers
// window.infoWindow = null; // Now initialized inside initializeMapWithMarkers
// window.markerClustererInstance = null; // Now initialized inside initializeMapWithMarkers
// window.markerClustererLoaded = false; // For tracking MarkerClusterer script load

console.log('Google Maps initialization script parsed and global functions defined.');

// ... (rest of your original file, if any, ensure it's compatible with these changes)
// For example, the handleMapData and MutationObserver part seems to be for a different mechanism
// It might conflict or be redundant now. Review if it's still needed.
// If it was for observing changes to a data div, the clientside_callback now handles that.

/*
  // Example of a more complex info window
  // ... (your original complex info window example) ...

  // The MutationObserver part - consider if this is still needed with clientside_callbacks
  // function handleMapData(mutationsList, observer) {
  // ...
  // }
  // const mapDataElement = document.getElementById('google-price-map-data');
  // if (mapDataElement) {
  //   const observer = new MutationObserver(handleMapData);
  //   observer.observe(mapDataElement, { childList: true, characterData: true, subtree: true });
  //   // Initial call if data is already present
  //   if (mapDataElement.textContent && mapDataElement.textContent.trim().length > 0) {
  //     try {
  //       const initialData = JSON.parse(mapDataElement.textContent);
  //       initGoogleMap(initialData, 'google-price-map-container');
  //     } catch (e) {
  //       console.error("Error parsing initial data from google-price-map-data:", e);
  //     }
  //   }
  // } else {
  //   console.error("Element with ID 'google-price-map-data' not found for MutationObserver.");
  // }
*/