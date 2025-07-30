'use client';
import { useEffect, useRef, useState } from 'react';
import { useMap } from '@/contexts/MapContext';
import { useFetchGeoJsonLayer } from '@/components/analytics/data';
import { MapGeoJSONFeature } from 'maplibre-gl';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

export default function H3Layer(props: {
  h3_index: any;
  iddesa: any;
  idkec: any;
  idkab: any;
  onH3Click: (h3_index: string) => void
}) {
  const { map, loaded } = useMap();

  const hoveredId = useRef<number | string | undefined | null>(null);
  const clickedId = useRef<number | string | undefined | null>(null);

  const loadGeoJSON = async () => {
    let params = new URLSearchParams({
      h3_index: props.h3_index,
      iddesa: props.iddesa,
      idkec: props.idkec,
      idkab: props.idkab,
    });
    const res = await useFetchGeoJsonLayer(params);

    const sourceId = 'pg-geojson';
    const layerId = 'pg-layer';

    // Remove old layer/source
    if (map && map.getLayer(layerId)) map.removeLayer(layerId);
    if (map && map.getSource(sourceId)) map.removeSource(sourceId);
    // if (map && map.getSource(bordersourceId)) map.removeSource(bordersourceId);

    map.addSource(sourceId, {
      type: 'geojson',
      data: res.data,
    });

    map.addLayer({
      id: layerId,
      type: 'fill',
      source: sourceId,
      paint: {
        'fill-color': [
          'case',
          ['boolean', ['feature-state', 'clicked'], false],
          '#991B1B', // dark red on click
          ['boolean', ['feature-state', 'hover'], false],
          '#0000FF', // yellow on hover
          [
            // risk palette
            'interpolate',
            ['linear'],
            ['get', 'anomaly_score_probability'],
            0.0,
            '#16A34A',
            0.2,
            '#84CC16',
            0.4,
            '#FACC15',
            0.6,
            '#FB923C',
            0.8,
            '#DC2626',
          ],
        ],
        'fill-opacity': 0.7,
      },
    });

    // Hover interaction
    map.on('mousemove', layerId, (e) => {
      const features = map.queryRenderedFeatures(e.point, {
        layers: [layerId],
      });
      // if (features?.length) return;

      const feature = features[0] as MapGeoJSONFeature;
      const id = feature.id as number;

      if (hoveredId.current !== null && hoveredId.current !== id) {
        map.setFeatureState(
          { source: sourceId, id: hoveredId.current },
          { hover: false },
        );
      }

      map.setFeatureState({ source: sourceId, id }, { hover: true });
      hoveredId.current = id;
      map.getCanvas().style.cursor = 'pointer';
    });

    map.on('mouseleave', layerId, () => {
      if (hoveredId.current !== null) {
        map.setFeatureState(
          { source: sourceId, id: hoveredId.current },
          { hover: false },
        );
        hoveredId.current = null;
      }
      map.getCanvas().style.cursor = '';
    });

    // Click interaction
    map.on('click', layerId, (e) => {
      const features = map.queryRenderedFeatures(e.point, {
        layers: [layerId],
      });
      const feature = features[0] as MapGeoJSONFeature;
      
      const id = feature.id as number;
      if (clickedId !== null && clickedId.current !== id) {
        map.setFeatureState(
          { source: sourceId, id: clickedId.current! },
          { clicked: false },
        );
      }
      map.setFeatureState({ source: sourceId, id }, { clicked: true });
      clickedId.current = id;
      // Show popup
      new maplibregl.Popup()
        .setLngLat(e.lngLat)
        .setHTML(
          `<strong>Kode:</strong> ${id}<br/><strong>Score:</strong> ${feature.properties.anomaly_score_probability}`,
        )
        .addTo(map)
        .on('close', () => {
          map.setFeatureState({ source: sourceId, id }, { clicked: false });
        });
      props.onH3Click(feature.properties.h3_index);
    });
  };

  useEffect(() => {
    if (!map || !loaded) return;

    if (map.isStyleLoaded()) {
      loadGeoJSON();
    } else {
      map.once('style.load', loadGeoJSON);
    }
  }, [map, props, loaded]);

  return null;
}
