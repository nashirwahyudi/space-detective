`use-client`;

import {
  Box,
  Button,
  Image,
  Flex,
  Select,
  Spinner,
  Center,
} from '@chakra-ui/react';
import { useEffect, useState } from 'react';
import { useFecthShap } from '@/components/analytics/data';

export default function AnomalyTable() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);
  // Filters
  const [shapFilter, setShapFilter] = useState('');

  //   options
  const [shapOptions, setShapOptions] = useState([]);

  const base_url =
    'https://ppatk-trackaml.oss-ap-southeast-5.aliyuncs.com/png/waterfall_plot_374.png';

  const fetchShaps = async () => {
    setLoading(true);
    try {
      let response = await useFecthShap();
      if (response.success) {
        setShapOptions(response.data);
        setShapFilter(response.data[0].png_relative_path);
      } else {
        throw Error(response.message);
      }
      setLoading(false);
    } catch (err: any) {
      setLoading(false);
      console.log(err);
    }
  };
  useEffect(() => {
    fetchShaps();
  }, []);

  return (
    <Box p={0}>
      <Flex gap="2" mb="4" wrap="wrap" direction="row">
        <Select
          placeholder="Pilih Index H3"
          // value={shapFilter}
          onChange={(e) => setShapFilter(e.target.value)}
          size="sm"
          w={'100%'}
          disabled={shapOptions.length == 0}
          defaultValue={shapFilter}
        >
          {shapOptions.map((elem: any, index) => (
            <option
              value={elem.png_relative_path}
              key={`${elem.h3_index}_${index}`}
            >
              {elem.h3_index}
            </option>
          ))}
        </Select>
      </Flex>

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
