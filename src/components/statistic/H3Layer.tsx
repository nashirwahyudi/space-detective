'use client';
import { useEffect } from 'react';
import { useMap } from '@/contexts/MapContext';
import { useFetchGeoJsonLayer } from '@/components/analytics/data';

export default function H3Layer(props: {
  h3_index: any;
  iddesa: any;
  idkec: any;
  idkab: any;
}) {
  const { map } = useMap();

  useEffect(() => {
    if (!map) return;

    const sourceId = 'pg-geojson';
    const layerId = 'pg-layer';
    const bordersourceId = 'pg-geojson-1';
    const borderlayerId = 'pg-layer-1';

    const loadGeoJSON = async () => {
      let params = new URLSearchParams({
        h3_index: props.h3_index,
        iddesa: props.iddesa,
        idkec: props.idkec,
        idkab: props.idkab,
      });
      const res = await useFetchGeoJsonLayer(params);

      // Remove old layer/source
      if (map && map.getLayer(layerId)) map.removeLayer(layerId);
      if (map && map.getSource(sourceId)) map.removeSource(sourceId);
      if (map && map.getLayer(borderlayerId)) map.removeLayer(borderlayerId);
      if (map && map.getSource(bordersourceId)) map.removeSource(bordersourceId);

      map.addSource(sourceId, {
        type: 'geojson',
        data: res.data,
      });
      map.addLayer({
        id: layerId,
        type: 'fill',
        source: sourceId,
        paint: {
          'fill-color': '#F97316',
          'fill-opacity': 0.6,
        },
      });

      map.addLayer({
        id: borderlayerId,
        type: 'line',
        source: bordersourceId,
        paint: {
          'line-color': '#EF4444',
          'line-width': 1,
        },
      });
    };

    loadGeoJSON();
  }, [map, props]);

  return null;
}
