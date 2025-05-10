// Google Maps initialization helper
window.addEventListener('DOMContentLoaded', function() {
  // Create global variable for Map data
  window.googleMapData = null;
  console.log('Google Maps initialization script loaded');
  
  // Verify that the Google Maps API is available
  function checkGoogleMapsApi() {
    if (typeof google === 'undefined' || typeof google.maps === 'undefined') {
      console.error('Google Maps API not loaded');
      const mapContainer = document.getElementById('google-price-map-container');
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
  
  // Load MarkerClusterer library
  function loadMarkerClusterer(callback) {
    if (window.markerClustererLoaded) {
      callback();
      return;
    }
    
    try {
      // Check if MarkerClusterer is already available
      if (typeof markerClusterer !== 'undefined') {
        window.markerClustererLoaded = true;
        callback();
        return;
      }
      
      // Load the MarkerClusterer script dynamically
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
        // Continue without clustering
        callback();
      };
      document.head.appendChild(script);
    } catch (error) {
      console.error('Error loading MarkerClusterer:', error);
      // Continue without clustering
      callback();
    }
  }
  
  // Function to initialize map when data is available
  window.initGoogleMap = function(data) {
    console.log('Initializing Google Maps with data');
    
    // Wait for container to be ready
    const waitForContainer = setInterval(function() {
      const mapContainer = document.getElementById('google-price-map-container');
      if (mapContainer) {
        clearInterval(waitForContainer);
        
        // Load MarkerClusterer and then initialize the map
        loadMarkerClusterer(function() {
          initializeMapWithMarkers(data, mapContainer);
        });
      }
    }, 100);
  };
  
  // Function to initialize the map and add markers
  function initializeMapWithMarkers(data, mapContainer) {
    try {
      if (!checkGoogleMapsApi()) {
        return;
      }
      
      const mapData = typeof data === 'string' ? JSON.parse(data) : data;
      console.log("Map Data received:", mapData);
      
      // Check for errors in the data
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
      
      // Ensure map center has valid coordinates
      const center = mapData.center || { lat: 41.6, lng: -93.6 }; // Default to Des Moines if center is invalid
      // Validate center coordinates
      if (typeof center.lat !== 'number' || typeof center.lng !== 'number' || 
          isNaN(center.lat) || isNaN(center.lng)) {
        console.error('Invalid map center:', center);
        center.lat = 41.6;
        center.lng = -93.6; // Default to Des Moines
      }
      
      // MODIFIED: Always recreate the map for a fresh state
      // Clear existing map object if it exists
      if (window.googleMap) {
        // Clear existing markers
        if (window.mapMarkers && window.mapMarkers.length > 0) {
          for (let marker of window.mapMarkers) {
            marker.setMap(null);
          }
        }
        
        // Clear existing marker clusterer
        if (window.markerClustererInstance) {
          window.markerClustererInstance = null;
        }
        
        // Log that we're completely recreating the map
        console.log(`Completely reinitializing map for new filtered data (timestamp: ${mapData.timestamp || 'none'})`);
        
        // Create a new map instance (this forces a complete refresh)
        window.googleMap = new google.maps.Map(mapContainer, {
          center: center,
          zoom: mapData.zoom || 12,
          mapTypeId: google.maps.MapTypeId.ROADMAP,
          mapTypeControl: true,
          streetViewControl: true,
          fullscreenControl: true
        });
        
        // Create new info window
        window.infoWindow = new google.maps.InfoWindow();
      } else {
        // Initialize map for the first time
        console.log(`Creating new Google Map centered at: ${center.lat}, ${center.lng}`);
        
        window.googleMap = new google.maps.Map(mapContainer, {
          center: center,
          zoom: mapData.zoom || 12,
          mapTypeId: google.maps.MapTypeId.ROADMAP,
          mapTypeControl: true,
          streetViewControl: true,
          fullscreenControl: true
        });
        
        // Create info window for markers
        window.infoWindow = new google.maps.InfoWindow();
      }
      
      // Store new markers
      window.mapMarkers = [];
      
      // Add markers for properties
      if (mapData.data && Array.isArray(mapData.data) && mapData.data.length > 0) {
        // Get price range from data statistics if available, otherwise calculate
        let minPrice, maxPrice;
        if (mapData.stats && typeof mapData.stats.price_min !== 'undefined' && 
            typeof mapData.stats.price_max !== 'undefined') {
          minPrice = mapData.stats.price_min;
          maxPrice = mapData.stats.price_max;
          console.log(`Using provided price range: $${minPrice} - $${maxPrice}`);
        } else {
          // Calculate price range from data points
          // Ensure we only use valid price values
          const prices = mapData.data
            .filter(item => typeof item.Sale_Price === 'number' && !isNaN(item.Sale_Price))
            .map(item => item.Sale_Price);
          
          if (prices.length === 0) {
            console.error('No valid price data found');
            return;
          }
          
          minPrice = Math.min(...prices);
          maxPrice = Math.max(...prices);
          console.log(`Calculated price range: $${minPrice} - $${maxPrice}`);
        }
        
        const priceRange = maxPrice - minPrice || 1; // Prevent division by zero
        
        console.log(`Adding ${mapData.data.length} markers to map`);
        
        // Debug the first few data points
        if (mapData.data.length > 0) {
          console.log("First marker data:", mapData.data[0]);
        }
        
        // Validate data points before adding markers
        let validMarkers = 0;
        let invalidMarkers = 0;
        
        // Add markers for each property
        for (let property of mapData.data) {
          // Skip invalid data points - thorough validation
          if (property === null || typeof property !== 'object') {
            console.warn('Skipping null or non-object property');
            invalidMarkers++;
            continue;
          }
          
          // Validate latitude, longitude and price
          const lat = parseFloat(property.Latitude);
          const lng = parseFloat(property.Longitude);
          const price = parseFloat(property.Sale_Price);
          
          if (isNaN(lat) || isNaN(lng) || isNaN(price) || 
              !isFinite(lat) || !isFinite(lng) || !isFinite(price)) {
            console.warn('Skipping invalid property data:', 
              {lat: property.Latitude, lng: property.Longitude, price: property.Sale_Price});
            invalidMarkers++;
            continue;
          }
          
          // Validate coordinate range
          if (lat < -90 || lat > 90 || lng < -180 || lng > 180) {
            console.warn('Skipping out-of-range coordinates:', {lat, lng});
            invalidMarkers++;
            continue;
          }
          
          const normalizedPrice = (price - minPrice) / priceRange;
          
          // Color gradient from green (low) to red (high)
          const hue = (1 - normalizedPrice) * 120;
          const color = `hsl(${hue}, 100%, 50%)`;
          
          // Create marker for this property
          const marker = new google.maps.Marker({
            position: {
              lat: lat,
              lng: lng
            },
            // Don't set the map yet if we're using clustering
            map: mapData.use_clustering ? null : window.googleMap,
            title: `$${price.toLocaleString()}`,
            icon: {
              path: google.maps.SymbolPath.CIRCLE,
              fillColor: color,
              fillOpacity: 0.7,
              strokeColor: 'white',
              strokeWeight: 1,
              scale: 8 + (normalizedPrice * 6) // Slightly smaller scale for better clustering
            }
          });
          
          // Create rich info window content
          let contentString = `
            <div style="padding: 8px; max-width: 300px;">
              <h5 style="margin-top: 0; color: #2c3e50;">$${price.toLocaleString()}</h5>
          `;
          
          // Add additional property details if available
          if (property.Bldg_Type) {
            contentString += `<p><b>Type:</b> ${property.Bldg_Type_Display || property.Bldg_Type}</p>`;
          }
          
          if (property.Year_Built) {
            contentString += `<p><b>Year Built:</b> ${property.Year_Built}</p>`;
          }
          
          if (property.Lot_Area) {
            contentString += `<p><b>Lot Area:</b> ${property.Lot_Area.toLocaleString()} sq.ft</p>`;
          }
          
          if (property.Neighborhood) {
            contentString += `<p><b>Neighborhood:</b> ${property.Neighborhood}</p>`;
          }
          
          contentString += `
              <p><b>Location:</b> ${lat.toFixed(6)}, ${lng.toFixed(6)}</p>
            </div>
          `;
          
          // Add click event listener
          marker.addListener('click', () => {
            window.infoWindow.setContent(contentString);
            window.infoWindow.open(window.googleMap, marker);
            
            // Store click data in the hidden div
            const clickData = {
              id: property.id || 0,
              lat: lat,
              lng: lng,
              price: price
            };
            
            const clickDataElement = document.getElementById('google-price-map-click-data');
            if (clickDataElement) {
              clickDataElement.textContent = JSON.stringify(clickData);
              
              // Dispatch a custom event that can be used by other callbacks
              const event = new CustomEvent('map-marker-click', { detail: clickData });
              document.dispatchEvent(event);
            }
          });
          
          window.mapMarkers.push(marker);
          validMarkers++;
        }
        
        console.log(`Marker validation summary: ${validMarkers} valid, ${invalidMarkers} invalid`);
        
        if (validMarkers === 0) {
          console.error('No valid markers could be created');
          mapContainer.innerHTML += `<div class="alert alert-warning" style="position: absolute; top: 10px; left: 10px; z-index: 1000; width: 80%;">
            No valid property coordinates found. Please check your data or try a different filter.
          </div>`;
          return;
        }
        
        // Apply marker clustering if enabled
        if (mapData.use_clustering && window.markerClustererLoaded) {
          try {
            // Check if the MarkerClusterer library is available
            if (typeof markerClusterer !== 'undefined') {
              // Create a new MarkerClusterer instance
              window.markerClustererInstance = new markerClusterer.MarkerClusterer({
                map: window.googleMap,
                markers: window.mapMarkers,
                algorithm: new markerClusterer.SuperClusterAlgorithm({
                  radius: 100, // Cluster radius in pixels
                  maxZoom: 16  // Max zoom level for clustering
                }),
                renderer: {
                  render: ({ count, position }) => {
                    return new google.maps.Marker({
                      position,
                      label: { text: String(count), color: "white", fontSize: "12px" },
                      icon: {
                        path: google.maps.SymbolPath.CIRCLE,
                        fillColor: "#4285F4",
                        fillOpacity: 0.9,
                        strokeWeight: 2,
                        strokeColor: "#FFFFFF",
                        scale: Math.min(count * 3, 30) // Size based on count
                      },
                      zIndex: Number(google.maps.Marker.MAX_ZINDEX) + count,
                    });
                  }
                }
              });
              
              console.log('Marker clustering applied');
            } else {
              console.warn('MarkerClusterer not available, showing markers without clustering');
              // Fall back to showing all markers
              window.mapMarkers.forEach(marker => marker.setMap(window.googleMap));
            }
          } catch (error) {
            console.error('Error applying marker clustering:', error);
            // Fall back to showing all markers
            window.mapMarkers.forEach(marker => marker.setMap(window.googleMap));
          }
        } else {
          // Set all markers to be visible on the map if not using clustering
          window.mapMarkers.forEach(marker => marker.setMap(window.googleMap));
        }
        
        // Create or update the legend
        if (window.mapLegend) {
          try {
            window.googleMap.controls[google.maps.ControlPosition.RIGHT_BOTTOM].removeAt(0);
          } catch (e) {
            console.warn('Error removing legend:', e);
          }
        }
        
        const legend = document.createElement('div');
        legend.style.backgroundColor = 'white';
        legend.style.padding = '10px';
        legend.style.margin = '10px';
        legend.style.borderRadius = '5px';
        legend.style.boxShadow = '0 2px 6px rgba(0,0,0,.3)';
        
        const title = document.createElement('h4');
        title.textContent = 'Price Legend';
        title.style.margin = '0 0 8px';
        legend.appendChild(title);
        
        // Add legend items
        const steps = 5;
        for (let i = 0; i < steps; i++) {
          const item = document.createElement('div');
          item.style.display = 'flex';
          item.style.alignItems = 'center';
          item.style.marginBottom = '5px';
          
          const normalizedValue = i / (steps - 1);
          const hue = (1 - normalizedValue) * 120;
          const color = `hsl(${hue}, 100%, 50%)`;
          
          const colorBox = document.createElement('div');
          colorBox.style.width = '20px';
          colorBox.style.height = '20px';
          colorBox.style.backgroundColor = color;
          colorBox.style.marginRight = '8px';
          colorBox.style.border = '1px solid #ccc';
          item.appendChild(colorBox);
          
          const price = minPrice + (priceRange * normalizedValue);
          const label = document.createElement('span');
          label.textContent = `$${Math.round(price).toLocaleString()}`;
          item.appendChild(label);
          
          legend.appendChild(item);
        }
        
        // Add count indicator with sampling information
        const count = document.createElement('div');
        count.style.marginTop = '10px';
        count.style.fontSize = '12px';
        count.style.color = '#666';
        
        // Show sampling information if available
        if (mapData.stats && mapData.stats.sampled && mapData.stats.total_count) {
          count.textContent = `${mapData.data.length} properties shown (sampled from ${mapData.stats.total_count} total)`;
          
          // Add a small note about clustering if applied
          if (mapData.stats.clustering_applied) {
            const clusterNote = document.createElement('div');
            clusterNote.style.fontSize = '11px';
            clusterNote.style.marginTop = '3px';
            clusterNote.textContent = 'Geographic clustering applied for better representation';
            count.appendChild(clusterNote);
          }
        } else {
          count.textContent = `${mapData.data.length} properties shown`;
        }
        
        legend.appendChild(count);
        
        window.mapLegend = legend;
        window.googleMap.controls[google.maps.ControlPosition.RIGHT_BOTTOM].push(legend);
        
        // Update the debug div with successful status
        const debugElement = document.getElementById('google-price-map-debug');
        if (debugElement) {
          debugElement.textContent = `Map loaded with ${mapData.data.length} properties`;
          debugElement.style.display = 'block';
        }
        
        // Add a loading indicator hiding animation
        const loadingElement = document.getElementById('google-price-map-loading-placeholder');
        if (loadingElement) {
          loadingElement.style.display = 'none';
        }
      } else {
        console.warn('No map data points available');
        // Check if data is actually empty or just not properly formatted
        console.log("mapData.data details:", { 
          exists: !!mapData.data, 
          isArray: Array.isArray(mapData.data), 
          length: mapData.data ? mapData.data.length : 0 
        });
        
        mapContainer.innerHTML += `<div class="alert alert-warning" style="position: absolute; top: 10px; left: 10px; z-index: 1000; width: 80%;">
          No property data available for the current filter criteria. Try changing your filters.
        </div>`;
      }
      
      console.log('Map initialized successfully');
      if (window.googleMap) {
        google.maps.event.trigger(window.googleMap, 'resize');
        console.log('Manually triggered map resize event.');
      }
    } catch (error) {
      console.error('Error initializing Google Maps:', error);
      mapContainer.innerHTML = `<div class="alert alert-danger">Error initializing map: ${error.message}</div>`;
    }
  }
  
  // Callback function to execute when mutations are observed
  function handleMapData(mutationsList, observer) {
    for (const mutation of mutationsList) {
        let relevantMutationDetected = false;
        let mutationDescription = "";

        if (mutation.type === 'childList' && mutation.target.id === 'google-price-map-data') {
            relevantMutationDetected = true;
            mutationDescription = `childList mutation on #${mutation.target.id}`;
        } else if (mutation.type === 'characterData' && mutation.target.parentElement && mutation.target.parentElement.id === 'google-price-map-data') {
            relevantMutationDetected = true;
            mutationDescription = `characterData mutation for #${mutation.target.parentElement.id}`;
        }

        if (relevantMutationDetected) {
            console.log(`MAP_JS_DEBUG: handleMapData triggered by ${mutationDescription}`);
            const dataElement = document.getElementById('google-price-map-data');
            
            if (dataElement && dataElement.textContent && dataElement.textContent.trim() !== '') {
                const jsonDataToParse = dataElement.textContent;
                console.log(`Map data received, length: ${jsonDataToParse.length} chars. Content (first 100): ${jsonDataToParse.substring(0,100)}`);
                try {
                    const parsedData = JSON.parse(jsonDataToParse);
                    console.log(`MAP_JS_DEBUG: Successfully parsed data. filter_change: ${parsedData.filter_change}, timestamp: ${parsedData.timestamp}, tab_activated: ${parsedData.tab_activated_timestamp}, data items: ${parsedData.data ? parsedData.data.length : 'N/A'}`);

                    // Always update the global data store first
                    window.googleMapData = parsedData; 

                    let callInitMap = false;
                    let reason = [];

                    if (parsedData.filter_change) {
                        console.log("Filter change detected! Scheduling map refresh with current data.");
                        callInitMap = true;
                        reason.push("filter_change");
                    }

                    if (parsedData.tab_activated_timestamp) {
                        console.log("Tab activation detected! Scheduling map refresh with current data if not already from filter_change.");
                        callInitMap = true; // Ensure map init if tab becomes active
                        if (!reason.includes("filter_change")) reason.push("tab_activated");
                    }

                    if (callInitMap && parsedData && parsedData.data && parsedData.center) {
                        if (checkGoogleMapsApi()) {
                            // Adding a slightly longer delay to see if it helps with any race conditions during rapid updates.
                            setTimeout(function() {
                                console.log(`Executing initGoogleMap (delay: 150ms). Reason: ${reason.join(', ')}. Using latest parsedData.`);
                                loadMarkerClusterer(function() { 
                                    // IMPORTANT: Use the parsedData that triggered this specific call
                                    window.initGoogleMap(parsedData); 
                                });
                            }, 150); 
                        } else {
                            console.warn("Google Maps API not ready, delaying map initialization.");
                        }
                    } else if (parsedData && parsedData.data === "FILTER_TRIGGER_DATA") {
                        console.warn("Received FILTER_TRIGGER_DATA. This should have been removed from Python. Map will likely be empty or incorrect.");
                    } else if (!callInitMap) {
                        console.log("MAP_JS_DEBUG: Conditions for calling initGoogleMap not met (no filter_change or tab_activated_timestamp). Parsed data:", parsedData);
                    }
                    else {
                        console.warn("Received map data is invalid, incomplete, or has no data/center. Not updating map.", parsedData);
                    }
                } catch (e) {
                     console.error('MAP_JS_DEBUG: Error parsing map data JSON. Raw content that failed (first 500 chars):', jsonDataToParse.substring(0, 500), 'Error:', e);
                }
                return; // Process only the first relevant mutation in the batch
            } else {
                console.warn("MAP_JS_DEBUG: Relevant mutation detected, but dataElement or its textContent is empty/null or whitespace.");
            }
        }
    }
  }
  
  // Instantiate the observer before using it in waitForDataElement
  const observer = new MutationObserver(handleMapData);

  // Wait for the map data element to exist, then start observing it
  const waitForDataElement = setInterval(function() {
    const dataElement = document.getElementById('google-price-map-data');
    if (dataElement) {
      clearInterval(waitForDataElement);
      console.log('Found google-price-map-data element');
      
      // Check if it already has data
      if (dataElement.textContent) {
        console.log('Initial data found, initializing map');
        // Call handleMapData directly IF initial content exists, to ensure consistent processing
        // Create a mock mutation list for the initial load, as handleMapData expects it
        const mockMutation = { type: 'childList', target: dataElement };
        handleMapData([mockMutation], observer); // Pass observer instance too
      } else {
        console.log('No initial data, waiting for updates');
      }
      
      // Observe future changes
      observer.observe(dataElement, { childList: true, characterData: true, subtree: true });
    }
  }, 100);
  
  // Check API status after a delay to ensure the API has time to load
  setTimeout(checkGoogleMapsApi, 2000);
});