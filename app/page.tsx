'use client';
/*eslint-disable*/

// import Map from '@/components/statistic/Map';
import MapProvider from '@/lib/maplibre/provider';
import { Flex, Box, useColorModeValue, Card } from '@chakra-ui/react';
import { useEffect, useRef, useState } from 'react';
import Table from '@/components/statistic/Table';
import Shap from '@/components/statistic/Shap';
import H3Layer from '@/components/statistic/H3Layer';
import { MapLegend } from '@/components/statistic/MapLegend';
import GlobalFilter from '@/components/statistic/GlobalFilter';

// export default function Chat(props: { apiKeyApp: string }) {
export default function Dashboard() {
  // reg
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  // Input States
  const textColor = useColorModeValue('navy.700', 'white');

  const [h3IndexClicked, setH3IndexClicked] = useState<string>('');
  // global filter
  const [filter, setFilter] = useState({
    h3_index: '',
    iddesa: '',
    idkec: '',
    idkab: '',
  });

  return (
    <Flex
      w="100%"
      pt={{ base: '70px', md: '0px' }}
      direction="column"
      position="relative"
    >
      <Flex
        direction="row"
        mx="auto"
        w={{ base: '100%', md: '100%', xl: '100%' }}
        maxW="1000px"
        mb={5}
      >
        <Flex direction={'row'} w="100%" h={{ minH: '5vh' }} mb={'auto'}>
          <Card
            display={'flex'}
            px="22px !important"
            py="22px !important"
            w={'100%'}
            color={textColor}
            fontSize={{ base: 'sm', md: 'md' }}
            lineHeight={{ base: '24px', md: '26px' }}
            fontWeight="500"
          >
            <GlobalFilter
              onFilterSubmit={(
                h3_index: string,
                iddesa: string,
                idkec: string,
                idkab: string,
              ) => {
                setFilter({
                  h3_index: h3_index,
                  iddesa: iddesa,
                  idkec: idkec,
                  idkab: idkab,
                });
                if (h3_index != '') {
                  setH3IndexClicked(h3_index);
                }
              }}
            />
          </Card>
        </Flex>
      </Flex>
      <Flex
        direction="row"
        mx="auto"
        w={{ base: '100%', md: '100%', xl: '100%' }}
        // minH={{ base: '75vh', '2xl': '85vh' }}
        maxW="1000px"
      >
        <Flex
          direction={'row'}
          w="50%"
          h={{ base: '85vh', '2xl': '85vh' }}
          mb={'auto'}
          marginEnd={5}
        >
          <Card
            display={'flex'}
            px="22px !important"
            py="22px !important"
            w={'100%'}
            h={'100%'}
            color={textColor}
            fontSize={{ base: 'sm', md: 'md' }}
            lineHeight={{ base: '24px', md: '26px' }}
            fontWeight="500"
          >
            {/* <Map/> */}
            <Box
              id="map-container"
              ref={mapContainerRef}
              w={'100%'}
              h={'100%'}
              display="absolute"
              inset={0}
            >
              <MapProvider
                mapContainerRef={mapContainerRef}
                initialViewState={{
                  longitude: 98.845901,
                  latitude: 2.916044,
                  zoom: 7,
                }}
              >
                <H3Layer {...filter} onH3Click={(h3_index) => {
                  setH3IndexClicked(h3_index);
                }}/>
                <MapLegend />
              </MapProvider>
            </Box>
          </Card>
        </Flex>
        <Flex
          direction={'column'}
          // h={{ base: '75vh', '2xl': '85vh' }}
          w="50%"
          mb={'auto'}
          gap="5"
        >
          <Card
            display={'flex'}
            px="22px !important"
            w={'100%'}
            py="22px !important"
            h={{ min: '50%' }}
            color={textColor}
            fontSize={{ base: 'sm', md: 'md' }}
            lineHeight={{ base: '24px', md: '26px' }}
            fontWeight="500"
          >
            <Table {...filter} />
          </Card>
          {(filter && filter.h3_index || h3IndexClicked) && (
            <Card
              display={'flex'}
              px="22px !important"
              w={'100%'}
              py="22px !important"
              h={{ min: '50%' }}
              color={textColor}
              fontSize={{ base: 'sm', md: 'md' }}
              lineHeight={{ base: '24px', md: '26px' }}
              fontWeight="500"
            >
              <Shap h3_index={h3IndexClicked} />
            </Card>
          )}
        </Flex>
      </Flex>
    </Flex>
  );
}
