`use-client`;

import { Box, Image, Flex, Select, Spinner, Center } from '@chakra-ui/react';
import { useEffect, useState } from 'react';
import { useFecthShap } from '../analytics/data';

export default function AnomalyTable(props: { h3_index: string | '' }) {
  const [loading, setLoading] = useState(false);
  // Filters
  const [shapFilter, setShapFilter] = useState('');

  const fetchImageData = async (h3_index: string) => {
    setLoading(true);
    const params = new URLSearchParams({
      h3Index: props.h3_index,
    });
    const resp = await useFecthShap(params);
    setLoading(false);

    try {
      if (resp.success) {
        setShapFilter(resp.data[0].png_relative_path);
      } else {
        throw Error(resp.message);
      }
    } catch (err: any) {}
  };

  useEffect(() => {
    if (props.h3_index != '') {
      fetchImageData(props.h3_index);
    }
  }, [props])

  return (
    <Box p={0}>
      <Flex gap="2" mb="4" wrap="wrap" direction="row"></Flex>

      <Flex wrap="wrap" mb="4">
        {loading ? (
          <Center w={'100%'}>
            <Spinner />
          </Center>
        ) : (
          <Image
            rounded="md"
            src={`https://ppatk-trackaml.oss-ap-southeast-5.aliyuncs.com/png/${shapFilter}`}
            alt=""
            w={'100%'}
          />
        )}
      </Flex>
    </Box>
  );
}
