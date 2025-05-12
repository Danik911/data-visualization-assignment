// Ensure dash_clientside object exists
window.dash_clientside = window.dash_clientside || {};

// Assign to the dash_clientside namespace
window.dash_clientside.googleMaps = {
    initMap: function(mapDataJSON, mapContainerId) {
        console.log('RENDER_DEBUG: dash_clientside.googleMaps.initMap CALLED. Container ID:', mapContainerId);

        if (!mapContainerId) {
            console.error('RENDER_DEBUG: mapContainerId is missing in initMap!');
            return ''; // Return empty string for the status message
        }
        const mapId = mapContainerId.replace('-container', ''); // Extract base map ID

        const mapElement = document.getElementById(mapContainerId);
        if (!mapElement) {
            console.error('RENDER_DEBUG: Map element not found for ID:', mapContainerId);
            return '';
        }

        // Ensure the Google Maps API is loaded
        if (typeof google === 'undefined' || typeof google.maps === 'undefined') {
            console.error('RENDER_DEBUG: Google Maps API not loaded.');
            // Attempt to load it if a callback `gm_authFailure` or similar is not already defined globally by the API script.
            // This is a fallback and might not always work depending on how the main API script is loaded.
            // It's generally better to ensure the API script in index_string is correctly loaded.
            return 'Google Maps API not ready';
        }
        
        let mapData;
        if (typeof mapDataJSON === 'string') {
            try {
                mapData = JSON.parse(mapDataJSON);
            } catch (e) {
                console.error('RENDER_DEBUG: Failed to parse mapDataJSON:', e, mapDataJSON);
                mapData = { data: [], center: { lat: 42.032974, lng: -93.61969 }, zoom: 10 }; // Default on error
            }
        } else {
            mapData = mapDataJSON; // Assume it's already an object
        }

        if (!mapData || !mapData.data) {
            console.warn('RENDER_DEBUG: mapData or mapData.data is null or undefined. Using default map settings.');
            // Provide default values if mapData or mapData.data is not available
            mapData = { data: [], center: { lat: 42.032974, lng: -93.61969 }, zoom: 10 };
        }
        
        console.log('RENDER_DEBUG: Parsed mapData:', mapData);

        const defaultCenter = { lat: 42.032974, lng: -93.61969 }; // Ames, Iowa
        const defaultZoom = 12;

        let currentCenter = mapData.center && mapData.center.lat && mapData.center.lng ? mapData.center : defaultCenter;
        let currentZoom = mapData.zoom !== undefined ? mapData.zoom : defaultZoom;
        
        // Initialize the map
        // Check if a map instance already exists for this container to avoid re-initializing
        // This basic check might need to be more robust depending on how Dash re-renders
        if (!mapElement.hasOwnProperty('_googleMapInstance')) {
            console.log('RENDER_DEBUG: Initializing new Google Map instance.');
            mapElement._googleMapInstance = new google.maps.Map(mapElement, {
                center: currentCenter,
                zoom: currentZoom,
                mapTypeId: 'roadmap',
                streetViewControl: false,
                mapTypeControl: true,
                mapTypeControlOptions: {
                    style: google.maps.MapTypeControlStyle.DROPDOWN_MENU,
                    position: google.maps.ControlPosition.TOP_RIGHT,
                },
                zoomControl: true,
                zoomControlOptions: {
                    position: google.maps.ControlPosition.LEFT_BOTTOM,
                },
                fullscreenControl: false,
            });
        } else {
            console.log('RENDER_DEBUG: Using existing Google Map instance. Setting center and zoom.');
            // If map already exists, just update its center and zoom if necessary
             mapElement._googleMapInstance.setCenter(currentCenter);
             mapElement._googleMapInstance.setZoom(currentZoom);
        }
        
        const map = mapElement._googleMapInstance;

        // Clear existing markers (if any) - simple approach: store markers in an array on the mapElement
        if (mapElement._markers && mapElement._markers.length > 0) {
            console.log(`RENDER_DEBUG: Clearing ${mapElement._markers.length} existing markers.`);
            mapElement._markers.forEach(marker => marker.setMap(null));
            mapElement._markers = [];
        } else {
            mapElement._markers = [];
        }
        
        let infoWindow = mapElement._infoWindow || new google.maps.InfoWindow();
        mapElement._infoWindow = infoWindow;


        if (mapData.data && mapData.data.length > 0) {
            console.log(`RENDER_DEBUG: Adding ${mapData.data.length} new markers.`);
            mapData.data.forEach(property => {
                if (property.Latitude != null && property.Longitude != null) {
                    const marker = new google.maps.Marker({
                        position: { lat: property.Latitude, lng: property.Longitude },
                        map: map,
                        title: 'Price: $' + property.Sale_Price + '\\nType: ' + property.Bldg_Type_Display,
                    });

                    marker.addListener('click', () => {
                        // Improved InfoWindow Content
                        const content = 
                            '<div style="font-family: Arial, sans-serif; padding: 5px; max-width: 250px;">' +
                                '<h6 style="margin: 0 0 10px 0; font-size: 1.1em; color: #333; border-bottom: 1px solid #eee; padding-bottom: 5px;">' +
                                    '<i class="fas fa-map-marker-alt" style="margin-right: 5px; color: #007bff;"></i>Property Details' +
                                '</h6>' +
                                '<table style="width: 100%; border-collapse: collapse;">' +
                                    '<tr>' +
                                        '<td style="padding: 4px 0; font-weight: bold; color: #555; width: 70px;"><i class="fas fa-dollar-sign" style="margin-right: 5px; color: #28a745;"></i>Price:</td>' +
                                        '<td style="padding: 4px 0; color: #333;">$' + property.Sale_Price.toLocaleString() + '</td>' +
                                    '</tr>' +
                                    '<tr>' +
                                        '<td style="padding: 4px 0; font-weight: bold; color: #555;"><i class="fas fa-building" style="margin-right: 5px; color: #17a2b8;"></i>Type:</td>' +
                                        '<td style="padding: 4px 0; color: #333;">' + (property.Bldg_Type_Display || 'N/A') + '</td>' +
                                    '</tr>' +
                                    '<tr>' +
                                        '<td style="padding: 4px 0; font-weight: bold; color: #555;"><i class="fas fa-ruler-combined" style="margin-right: 5px; color: #ffc107;"></i>Area:</td>' +
                                        '<td style="padding: 4px 0; color: #333;">' + (property.Lot_Area ? property.Lot_Area.toLocaleString() + ' sqft' : 'N/A') + '</td>' +
                                    '</tr>' +
                                    '<tr>' +
                                        '<td style="padding: 4px 0; font-weight: bold; color: #555;"><i class="fas fa-calendar-alt" style="margin-right: 5px; color: #6f42c1;"></i>Built:</td>' +
                                        '<td style="padding: 4px 0; color: #333;">' + (property.Year_Built || 'N/A') + '</td>' +
                                    '</tr>' +
                                '</table>' +
                            '</div>';
                        infoWindow.setContent(content);
                        infoWindow.open(map, marker);
                    });
                    mapElement._markers.push(marker);
                } else {
                     console.warn('RENDER_DEBUG: Marker skipped due to null Latitude/Longitude:', property);
                }
            });
        } else {
            console.log('RENDER_DEBUG: No marker data provided or data is empty.');
        }

        const message = 'Map updated with ' + (mapData.data ? mapData.data.length : 0) + ' markers.';
        console.log('RENDER_DEBUG: initMap for ' + mapId + ' returning message: ' + message);
        return message; // This message goes to the google-price-map-status-message div
    },

    clearMessage: function(currentMessage) {
        // The input `currentMessage` is the content of the status div.
        // We need to know the ID of the status div to clear it.
        // Assuming the map ID is 'google-price-map' based on the Python code.
        // If map IDs can vary, this part needs to be more dynamic or the ID passed.
        const statusDivId = 'google-price-map-status-message'; 
        const statusDiv = document.getElementById(statusDivId);

        if (statusDiv && currentMessage) { // Only run if there's a message
            console.log('RENDER_DEBUG: clearMessage called for ' + statusDivId + ' with message: ' + currentMessage + '. Clearing in 3s.');
            setTimeout(() => {
                if (document.getElementById(statusDivId)) { // Check if element still exists
                    document.getElementById(statusDivId).innerText = '';
                    console.log('RENDER_DEBUG: Message cleared for ' + statusDivId + '.');
                }
            }, 3000); // Clear after 3 seconds
        }
        // This callback needs to return a value for its Output, but since we're directly manipulating DOM,
        // and allow_duplicate=True is set, returning the same message or an empty string is fine.
        // It's better to return dash.no_update if possible, but that's for server-side.
        // For clientside, we usually return a value that makes sense for the Output.
        // Here, we are modifying the output directly, so we don't want to cause another update cycle.
        // However, a clientside callback MUST return a value for its output.
        // Returning the original message won't re-trigger this specific callback if its Output value doesn't change.
        return currentMessage; 
    }
};

console.log('Google Maps init.js PARSED and dash_clientside.googleMaps defined. Includes clearMessage.');

// All other previous JS code from this file is temporarily removed for this test.