<?php

namespace Tests\YooKassa\Helpers;

use PHPUnit\Framework\TestCase;
use ReflectionClass;
use ReflectionException;
use YooKassa\Helpers\ProductCode;
use YooKassa\Model\Receipt\MarkCodeInfo;

/**
 * @internal
 */
class ProductCodeTest extends TestCase
{
    /**
     * @dataProvider dataProvider
     */
    public function testSetGetPrefix(mixed $data): void
    {
        $instance = new ProductCode();

        self::assertEquals(ProductCode::PREFIX_UNKNOWN, $instance->getPrefix());

        $instance->setPrefix($data['prefix']);
        if (empty($data['prefix'])) {
            self::assertNull($instance->getPrefix());
        } else {
            self::assertNotNull($instance->getPrefix());
        }
    }

    /**
     * @dataProvider dataProvider
     */
    public function testSetGetGtin(mixed $data): void
    {
        $instance = new ProductCode();

        $instance->setGtin($data['gtin']);
        if (empty($data['gtin'])) {
            self::assertNull($instance->getGtin());
        } else {
            self::assertEquals($data['gtin'], $instance->getGtin());
        }
    }

    /**
     * @dataProvider dataProvider
     */
    public function testSetGetUsePrefix(mixed $data): void
    {
        $instance = new ProductCode();

        self::assertTrue($instance->isUsePrefix());

        $instance->setUsePrefix($data['usePrefix']);
        if (empty($data['usePrefix'])) {
            self::assertFalse($instance->isUsePrefix());
        } else {
            self::assertTrue($instance->isUsePrefix());
        }
    }

    /**
     * @dataProvider dataProvider
     */
    public function testSetGetSerial(mixed $data): void
    {
        $instance = new ProductCode();

        $instance->setSerial($data['serial']);
        if (empty($data['serial'])) {
            self::assertNull($instance->getSerial());
        } else {
            self::assertEquals($data['serial'], $instance->getSerial());
        }
    }

    /**
     * @dataProvider dataProvider
     */
    public function testValidate(mixed $data): void
    {
        $instance = new ProductCode();

        self::assertFalse($instance->validate());

        $instance->setType($data['type']);
        $instance->setGtin($data['gtin']);
        $instance->setSerial($data['serial']);

        switch ($instance->getType()) {
            case ProductCode::TYPE_EAN_8:
            case ProductCode::TYPE_EAN_13:
            case ProductCode::TYPE_ITF_14:
            case ProductCode::TYPE_FUR:
            case ProductCode::TYPE_EGAIS_20:
            case ProductCode::TYPE_EGAIS_30:
            case ProductCode::TYPE_UNKNOWN:
                if (empty($data['gtin'])) {
                    self::assertFalse($instance->validate());
                }
                if (empty($data['gtin'])) {
                    self::assertTrue($instance->validate());
                }

                break;

            case ProductCode::TYPE_GS_10:
            case ProductCode::TYPE_GS_1M:
            case ProductCode::TYPE_SHORT:
                if (empty($data['gtin']) || empty($data['serial'])) {
                    self::assertFalse($instance->validate());
                }
                if (!empty($data['gtin']) || !empty($data['serial'])) {
                    self::assertTrue($instance->validate());
                }

                break;
        }
    }

    /**
     * @dataProvider dataStrProvider
     *
     * @throws ReflectionException
     */
    public function testStrHex(mixed $data): void
    {
        $instance = new ProductCode();
        $reflection = new ReflectionClass($instance::class);

        $method = $reflection->getMethod('strToHex');
        $method->setAccessible(true);
        $result1 = $method->invokeArgs($instance, ['string' => $data]);

        $method = $reflection->getMethod('hexToStr');
        $method->setAccessible(true);
        $result2 = $method->invokeArgs($instance, ['hex' => $result1]);

        self::assertEquals($data, $result2);
    }

    /**
     * @dataProvider dataNumProvider
     *
     * @throws ReflectionException
     */
    public function testNumHex(mixed $data): void
    {
        $instance = new ProductCode();
        $reflection = new ReflectionClass($instance::class);

        $method = $reflection->getMethod('baseConvert');
        $method->setAccessible(true);
        $result1 = $method->invokeArgs($instance, ['numString' => $data]);

        $method2 = $reflection->getMethod('baseConvert');
        $method2->setAccessible(true);
        $result2 = $method2->invokeArgs($instance, ['numString' => $result1, 'fromBase' => 16, 'toBase' => 10]);

        self::assertEquals(
            str_pad($data, 14, '0', STR_PAD_LEFT),
            str_pad($result2, 14, '0', STR_PAD_LEFT)
        );
    }

    /**
     * @dataProvider dataProvider
     */
    public function testGetResult(mixed $data): void
    {
        $instance = new ProductCode();

        self::assertEmpty($instance->getResult());

        $instance->setGtin($data['gtin']);
        $instance->setSerial($data['serial']);
        $instance->setPrefix($data['prefix']);
        $instance->setUsePrefix($data['usePrefix']);

        if ($instance->validate()) {
            self::assertNotEmpty($instance->calcResult());
            self::assertNotEmpty($instance->getResult());
        } else {
            self::assertEmpty($instance->calcResult());
            self::assertEmpty($instance->getResult());
        }
    }

    /**
     * @dataProvider dataProvider
     */
    public function testMagicMethods(mixed $data): void
    {
        $instance = new ProductCode($data['dataMatrix'], $data['usePrefix'] ? $data['prefix'] : false);

        if ($instance->validate()) {
            self::assertNotEmpty($instance->calcResult());
            self::assertNotEmpty($instance->getResult());
            self::assertEquals($data['result'], $instance->calcResult());
            self::assertEquals($data['result'], $instance->getResult());
            self::assertEquals($data['result'], (string) $instance);
        } else {
            self::assertEmpty($instance->getResult());
            self::assertEmpty($instance->getResult());
        }

        $instance2 = new ProductCode($data['dataMatrix'], $data['usePrefix'] ? $data['prefix'] : false);

        if ($instance2->validate()) {
            self::assertEquals($data['result'], (string) $instance2);
        } else {
            self::assertEmpty($instance2->getResult());
        }
    }

    /**
     * @dataProvider dataProvider
     */
    public function testGetSetMarkCodeInfo(array $data): void
    {
        $instance = new ProductCode($data['dataMatrix'], $data['usePrefix'] ? $data['prefix'] : false);

        if (empty($data['dataMatrix'])) {
            self::assertNull($instance->getMarkCodeInfo());

            return;
        }
        if (empty($data['mark_code_info'])) {
            self::assertInstanceOf(MarkCodeInfo::class, $instance->getMarkCodeInfo());
        } else {
            self::assertNotNull($instance->getMarkCodeInfo());
            if (!is_object($data['mark_code_info'])) {
                self::assertEquals($data['mark_code_info'], $instance->getMarkCodeInfo()->toArray());
            } else {
                self::assertEquals($data['mark_code_info']->toArray(), $instance->getMarkCodeInfo()->toArray());
            }
        }
    }

    public static function dataProvider(): array
    {
        return [
            [
                [
                    'usePrefix' => false,
                    'prefix' => null,
                    'type' => 'gs_1m',
                    'gtin' => '04630037591316',
                    'serial' => 'sgEKKPPcS25y5',
                    'dataMatrix' => '010463003407001221SxMGorvNuq6Wk91fgr92sdfsdfghfgjh',
                    'result' => '04 36 03 89 39 FC 53 78 4D 47 6F 72 76 4E 75 71 36 57 6B',
                ],
            ],
            [
                [
                    'usePrefix' => true,
                    'prefix' => true,
                    'type' => 'gs_1m',
                    'gtin' => '4630037591316',
                    'serial' => 'sgEKKPPcS25y5',
                    'dataMatrix' => '010463003407001221SxMGorvNuq6Wk91fgr92sdfsdfghfgjh',
                    'result' => '44 4D 04 36 03 89 39 FC 53 78 4D 47 6F 72 76 4E 75 71 36 57 6B',
                ],
            ],
            [
                [
                    'usePrefix' => true,
                    'prefix' => '444D',
                    'type' => 'gs_1m',
                    'gtin' => '04630037591316',
                    'serial' => 'sgEKKPPcS25y5',
                    'dataMatrix' => '010463003407001221SxMGorvNuq6Wk91fgr92sdfsdfghfgjh',
                    'result' => '44 4D 04 36 03 89 39 FC 53 78 4D 47 6F 72 76 4E 75 71 36 57 6B',
                ],
            ],
            [
                [
                    'usePrefix' => true,
                    'prefix' => 17485,
                    'type' => 'gs_1m',
                    'gtin' => '04630037591316',
                    'serial' => 'sgEKKPPcS25y5',
                    'dataMatrix' => '010463003407001221SxMGorvNuq6Wk91fgr92sdfsdfghfgjh',
                    'result' => '44 4D 04 36 03 89 39 FC 53 78 4D 47 6F 72 76 4E 75 71 36 57 6B',
                ],
            ],
            [
                [
                    'usePrefix' => false,
                    'prefix' => null,
                    'type' => 'gs_1m',
                    'gtin' => '98765432101234',
                    'serial' => 'ABC1234',
                    'dataMatrix' => '019876543210123421ABC123491DmUO92sdfJSf/"fgjh',
                    'result' => '59 D3 9E 7F 19 72 41 42 43 31 32 33 34',
                ],
            ],
            [
                [
                    'usePrefix' => false,
                    'prefix' => null,
                    'type' => 'gs_10',
                    'gtin' => '98765432101234',
                    'serial' => 'ABC1234',
                    'dataMatrix' => '019876543210123421ABC1234',
                    'result' => '59 D3 9E 7F 19 72 41 42 43 31 32 33 34',
                    'mark_code_info' => [
                        'gs_10' => '019876543210123421ABC1234',
                    ],
                ],
            ],
            [
                [
                    'usePrefix' => true,
                    'prefix' => '444D',
                    'type' => 'gs_1m',
                    'gtin' => '04650157546607',
                    'serial' => '1>iGl4Mmypg,w',
                    'dataMatrix' => '0104650157546607211>iGl4Mmypg,w91006492TCetfbrFRjPke+Scq0sUS9WX84GMz5zx+UR5y8I0QFCDte3E947cwYo3liObLS1uzTqOnzBkSjNJD+97P6OF7A==',
                    'result' => '44 4D 04 3A B2 FD 1C 6F 31 3E 69 47 6C 34 4D 6D 79 70 67 2C 77',
                    'mark_code_info' => [
                        'gs_1m' => '0104650157546607211>iGl4Mmypg,w',
                    ],
                ],
            ],
            [
                [
                    'usePrefix' => true,
                    'prefix' => '444D',
                    'type' => 'gs_1m',
                    'gtin' => '04601936005464',
                    'serial' => '5hHUGT',
                    'dataMatrix' => '0104601936005464215hHUGT93dpD0',
                    'result' => '44 4D 04 2F 78 C2 C9 58 35 68 48 55 47 54',
                    'mark_code_info' => new MarkCodeInfo([
                        'gs_1m' => '0104601936005464215hHUGT',
                    ]),
                ],
            ],
            [
                [
                    'usePrefix' => true,
                    'prefix' => '4909',
                    'type' => 'itf_14',
                    'gtin' => '01234567890123',
                    'serial' => '',
                    'dataMatrix' => '01234567890123',
                    'result' => '49 09 01 1F 71 FB 04 CB',
                    'mark_code_info' => [
                        'itf_14' => '01234567890123',
                    ],
                ],
            ],
            [
                [
                    'usePrefix' => true,
                    'prefix' => '450D',
                    'type' => 'ean_13',
                    'gtin' => '1234567890123',
                    'serial' => '',
                    'dataMatrix' => '1234567890123',
                    'result' => '45 0D 01 1F 71 FB 04 CB',
                    'mark_code_info' => [
                        'ean_13' => '1234567890123',
                    ],
                ],
            ],
            [
                [
                    'usePrefix' => true,
                    'prefix' => '4508',
                    'type' => 'ean_8',
                    'gtin' => '12345678',
                    'serial' => '',
                    'dataMatrix' => '12345678',
                    'result' => '45 08 00 00 00 BC 61 4E',
                    'mark_code_info' => [
                        'ean_8' => '12345678',
                    ],
                ],
            ],
            [
                [
                    'usePrefix' => true,
                    'prefix' => '5246',
                    'type' => 'fur',
                    'gtin' => 'AB-123456-ABCDEFGHIJ',
                    'serial' => '',
                    'dataMatrix' => 'AB-123456-ABCDEFGHIJ',
                    'result' => '52 46 41 42 2D 31 32 33 34 35 36 2D 41 42 43 44 45 46 47 48 49 4A',
                    'mark_code_info' => new MarkCodeInfo([
                        'fur' => 'AB-123456-ABCDEFGHIJ',
                    ]),
                ],
            ],
            [
                [
                    'usePrefix' => true,
                    'prefix' => '5246',
                    'type' => 'fur',
                    'gtin' => 'RU-430302-ABC1234567',
                    'serial' => '',
                    'dataMatrix' => 'RU-430302-ABC1234567',
                    'result' => '52 46 52 55 2D 34 33 30 33 30 32 2D 41 42 43 31 32 33 34 35 36 37',
                ],
            ],
            [
                [
                    'usePrefix' => true,
                    'prefix' => 'C514',
                    'type' => 'egais_20',
                    'gtin' => '52419RAYLTL37TX31219019',
                    'serial' => '',
                    'dataMatrix' => '20N0000152419RAYLTL37TX31219019004460R96J2Z9KXYLKRCWGQOUOIUDYXVY6MCW',
                    'result' => 'C5 14 35 32 34 31 39 52 41 59 4C 54 4C 33 37 54 58 33 31 32 31 39 30 31 39 30 30 34 34 36 30 52 39 36 4A',
                ],
            ],
            [
                [
                    'usePrefix' => true,
                    'prefix' => 'C51E',
                    'type' => 'egais_30',
                    'gtin' => '19530126773682',
                    'serial' => '',
                    'dataMatrix' => '195301267736820121001EO2BBV3RRHXBJTZYOD5A2IU5XMHT3ZKW44TYYAYE3GXPGSQTGJQS43TG5WLWTQVNHVHI2A6H7RBFYY4VO4THCK5HSFFUPMCAA7ZHNNXZ4ILMESLIGZPONDEQ5CK3J76JY',
                    'result' => 'C5 1E 31 39 35 33 30 31 32 36 37 37 33 36 38 32',
                ],
            ],
            [
                [
                    'usePrefix' => true,
                    'prefix' => '444D',
                    'type' => 'short',
                    'gtin' => '00046070061125',
                    'serial' => '75215MEZgB933==',
                    'dataMatrix' => '0004607006112575215MEZgB933==',
                    'result' => '44 4D 00 0A B9 FD 58 45 37 35 32 31 35 4D 45 5A 67 42 39 33 33 3D 3D',
                ],
            ],
            [
                [
                    'usePrefix' => true,
                    'prefix' => '444D',
                    'type' => 'gs_10',
                    'gtin' => '04630034070012',
                    'serial' => 'CMK45BrhN0WLf',
                    'dataMatrix' => '010463003407001221CMK45BrhN0WLf',
                    'result' => '44 4D 04 36 03 89 39 FC 43 4D 4B 34 35 42 72 68 4E 30 57 4C 66',
                ],
            ],
            [
                [
                    'usePrefix' => true,
                    'prefix' => '444D',
                    'type' => 'gs_10',
                    'gtin' => '04600439931256',
                    'serial' => 'JgXJ5.T',
                    'dataMatrix' => '010460043993125621JgXJ5.T\u001d8005112000\u001d930001\u001d923zbrLA==\u001d24014276281.',
                    'result' => '44 4D 04 2F 1F 96 81 78 4A 67 58 4A 35 2E 54 31 31 32 30 30 30',
                ],
            ],
            [
                [
                    'usePrefix' => true,
                    'prefix' => '0000',
                    'type' => 'unknown',
                    'gtin' => '123%123&sldkfjldksfj*(*',
                    'serial' => '123%123&sldkfjldksfj*(*',
                    'dataMatrix' => '123%123&sldkfjldksfj*(*',
                    'result' => '00 00 31 32 33 25 31 32 33 26 73 6C 64 6B 66 6A 6C 64 6B 73 66 6A 2A 28 2A',
                ],
            ],
            [
                [
                    'usePrefix' => false,
                    'prefix' => '',
                    'type' => '',
                    'gtin' => '',
                    'serial' => '',
                    'dataMatrix' => '',
                    'result' => '',
                ],
            ],
        ];
    }

    public static function dataStrProvider(): array
    {
        return [
            [
                'this is a test!',
            ],
            [
                'ABC1234',
            ],
            [
                'sgEKKPPcS25y5',
            ],
        ];
    }

    public static function dataNumProvider(): array
    {
        return [
            [
                '98765432101234',
            ],
            [
                '04630037591316',
            ],
            [
                '04603725748040',
            ],
            [
                '04603725748057',
            ],
            [
                '4603725748057',
            ],
        ];
    }
}
