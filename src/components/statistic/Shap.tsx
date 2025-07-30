`use-client`;

import { Box, Image, Flex, Select, Spinner, Center } from '@chakra-ui/react';
import { useEffect, useState } from 'react';

export default function AnomalyTable(props: { options: Array<any> }) {
  const [loading, setLoading] = useState(false);
  // Filters
  const [shapFilter, setShapFilter] = useState('');

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
