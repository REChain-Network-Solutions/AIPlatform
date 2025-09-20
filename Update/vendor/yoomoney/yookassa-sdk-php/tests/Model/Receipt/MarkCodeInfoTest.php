<?php

/*
* The MIT License
*
* Copyright (c) 2025 "YooMoney", NBСO LLC
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/

namespace Tests\YooKassa\Model\Receipt;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Receipt\MarkCodeInfo;

/**
 * MarkCodeInfoTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class MarkCodeInfoTest extends AbstractTestCase
{
    protected MarkCodeInfo $object;

    /**
     * @param mixed|null $value
     * @return MarkCodeInfo
     */
    protected function getTestInstance(mixed $value = null): MarkCodeInfo
    {
        return new MarkCodeInfo($value);
    }

    /**
     * @return void
     */
    public function testMarkCodeInfoClassExists(): void
    {
        $this->object = $this->getMockBuilder(MarkCodeInfo::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(MarkCodeInfo::class));
        $this->assertInstanceOf(MarkCodeInfo::class, $this->object);
    }

    /**
     * Test property "mark_code_raw"
     * @dataProvider validMarkCodeRawDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testMarkCodeRaw(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getMarkCodeRaw());
        self::assertEmpty($instance->mark_code_raw);
        $instance->setMarkCodeRaw($value);
        self::assertEquals($value, is_array($value) ? $instance->getMarkCodeRaw()->toArray() : $instance->getMarkCodeRaw());
        self::assertEquals($value, is_array($value) ? $instance->mark_code_raw->toArray() : $instance->mark_code_raw);
        if (!empty($value)) {
            self::assertNotNull($instance->getMarkCodeRaw());
            self::assertNotNull($instance->mark_code_raw);
        }
    }

    /**
     * Test invalid property "mark_code_raw"
     * @dataProvider invalidMarkCodeRawDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidMarkCodeRaw(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setMarkCodeRaw($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validMarkCodeRawDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_mark_code_raw'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidMarkCodeRawDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_mark_code_raw'));
    }

    /**
     * Test property "unknown"
     * @dataProvider validUnknownDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testUnknown(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getUnknown());
        self::assertEmpty($instance->unknown);
        $instance->setUnknown($value);
        self::assertEquals($value, is_array($value) ? $instance->getUnknown()->toArray() : $instance->getUnknown());
        self::assertEquals($value, is_array($value) ? $instance->unknown->toArray() : $instance->unknown);
        if (!empty($value)) {
            self::assertNotNull($instance->getUnknown());
            self::assertNotNull($instance->unknown);
            self::assertLessThanOrEqual(32, is_string($instance->getUnknown()) ? mb_strlen($instance->getUnknown()) : $instance->getUnknown());
            self::assertLessThanOrEqual(32, is_string($instance->unknown) ? mb_strlen($instance->unknown) : $instance->unknown);
            self::assertGreaterThanOrEqual(1, is_string($instance->getUnknown()) ? mb_strlen($instance->getUnknown()) : $instance->getUnknown());
            self::assertGreaterThanOrEqual(1, is_string($instance->unknown) ? mb_strlen($instance->unknown) : $instance->unknown);
        }
    }

    /**
     * Test invalid property "unknown"
     * @dataProvider invalidUnknownDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidUnknown(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setUnknown($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validUnknownDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_unknown'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidUnknownDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_unknown'));
    }

    /**
     * Test property "ean_8"
     * @dataProvider validEan8DataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testEan8(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getEan8());
        self::assertEmpty($instance->ean_8);
        $instance->setEan8($value);
        self::assertEquals($value, is_array($value) ? $instance->getEan8()->toArray() : $instance->getEan8());
        self::assertEquals($value, is_array($value) ? $instance->ean_8->toArray() : $instance->ean_8);
        if (!empty($value)) {
            self::assertNotNull($instance->getEan8());
            self::assertNotNull($instance->ean_8);
            self::assertLessThanOrEqual(8, is_string($instance->getEan8()) ? mb_strlen($instance->getEan8()) : $instance->getEan8());
            self::assertLessThanOrEqual(8, is_string($instance->ean_8) ? mb_strlen($instance->ean_8) : $instance->ean_8);
            self::assertGreaterThanOrEqual(8, is_string($instance->getEan8()) ? mb_strlen($instance->getEan8()) : $instance->getEan8());
            self::assertGreaterThanOrEqual(8, is_string($instance->ean_8) ? mb_strlen($instance->ean_8) : $instance->ean_8);
        }
    }

    /**
     * Test invalid property "ean_8"
     * @dataProvider invalidEan8DataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidEan8(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setEan8($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validEan8DataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_ean_8'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidEan8DataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_ean_8'));
    }

    /**
     * Test property "ean_13"
     * @dataProvider validEan13DataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testEan13(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getEan13());
        self::assertEmpty($instance->ean_13);
        $instance->setEan13($value);
        self::assertEquals($value, is_array($value) ? $instance->getEan13()->toArray() : $instance->getEan13());
        self::assertEquals($value, is_array($value) ? $instance->ean_13->toArray() : $instance->ean_13);
        if (!empty($value)) {
            self::assertNotNull($instance->getEan13());
            self::assertNotNull($instance->ean_13);
            self::assertLessThanOrEqual(13, is_string($instance->getEan13()) ? mb_strlen($instance->getEan13()) : $instance->getEan13());
            self::assertLessThanOrEqual(13, is_string($instance->ean_13) ? mb_strlen($instance->ean_13) : $instance->ean_13);
            self::assertGreaterThanOrEqual(13, is_string($instance->getEan13()) ? mb_strlen($instance->getEan13()) : $instance->getEan13());
            self::assertGreaterThanOrEqual(13, is_string($instance->ean_13) ? mb_strlen($instance->ean_13) : $instance->ean_13);
        }
    }

    /**
     * Test invalid property "ean_13"
     * @dataProvider invalidEan13DataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidEan13(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setEan13($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validEan13DataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_ean_13'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidEan13DataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_ean_13'));
    }

    /**
     * Test property "itf_14"
     * @dataProvider validItf14DataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testItf14(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getItf14());
        self::assertEmpty($instance->itf_14);
        $instance->setItf14($value);
        self::assertEquals($value, is_array($value) ? $instance->getItf14()->toArray() : $instance->getItf14());
        self::assertEquals($value, is_array($value) ? $instance->itf_14->toArray() : $instance->itf_14);
        if (!empty($value)) {
            self::assertNotNull($instance->getItf14());
            self::assertNotNull($instance->itf_14);
            self::assertLessThanOrEqual(14, is_string($instance->getItf14()) ? mb_strlen($instance->getItf14()) : $instance->getItf14());
            self::assertLessThanOrEqual(14, is_string($instance->itf_14) ? mb_strlen($instance->itf_14) : $instance->itf_14);
            self::assertGreaterThanOrEqual(14, is_string($instance->getItf14()) ? mb_strlen($instance->getItf14()) : $instance->getItf14());
            self::assertGreaterThanOrEqual(14, is_string($instance->itf_14) ? mb_strlen($instance->itf_14) : $instance->itf_14);
        }
    }

    /**
     * Test invalid property "itf_14"
     * @dataProvider invalidItf14DataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidItf14(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setItf14($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validItf14DataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_itf_14'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidItf14DataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_itf_14'));
    }

    /**
     * Test property "gs_10"
     * @dataProvider validGs10DataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testGs10(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getGs10());
        self::assertEmpty($instance->gs_10);
        $instance->setGs10($value);
        self::assertEquals($value, is_array($value) ? $instance->getGs10()->toArray() : $instance->getGs10());
        self::assertEquals($value, is_array($value) ? $instance->gs_10->toArray() : $instance->gs_10);
        if (!empty($value)) {
            self::assertNotNull($instance->getGs10());
            self::assertNotNull($instance->gs_10);
            self::assertLessThanOrEqual(38, is_string($instance->getGs10()) ? mb_strlen($instance->getGs10()) : $instance->getGs10());
            self::assertLessThanOrEqual(38, is_string($instance->gs_10) ? mb_strlen($instance->gs_10) : $instance->gs_10);
            self::assertGreaterThanOrEqual(1, is_string($instance->getGs10()) ? mb_strlen($instance->getGs10()) : $instance->getGs10());
            self::assertGreaterThanOrEqual(1, is_string($instance->gs_10) ? mb_strlen($instance->gs_10) : $instance->gs_10);
        }
    }

    /**
     * Test invalid property "gs_10"
     * @dataProvider invalidGs10DataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidGs10(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setGs10($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validGs10DataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_gs_10'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidGs10DataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_gs_10'));
    }

    /**
     * Test property "gs_1m"
     * @dataProvider validGs1mDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testGs1m(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getGs1m());
        self::assertEmpty($instance->gs_1m);
        $instance->setGs1m($value);
        self::assertEquals($value, is_array($value) ? $instance->getGs1m()->toArray() : $instance->getGs1m());
        self::assertEquals($value, is_array($value) ? $instance->gs_1m->toArray() : $instance->gs_1m);
        if (!empty($value)) {
            self::assertNotNull($instance->getGs1m());
            self::assertNotNull($instance->gs_1m);
            self::assertLessThanOrEqual(200, is_string($instance->getGs1m()) ? mb_strlen($instance->getGs1m()) : $instance->getGs1m());
            self::assertLessThanOrEqual(200, is_string($instance->gs_1m) ? mb_strlen($instance->gs_1m) : $instance->gs_1m);
            self::assertGreaterThanOrEqual(1, is_string($instance->getGs1m()) ? mb_strlen($instance->getGs1m()) : $instance->getGs1m());
            self::assertGreaterThanOrEqual(1, is_string($instance->gs_1m) ? mb_strlen($instance->gs_1m) : $instance->gs_1m);
        }
    }

    /**
     * Test invalid property "gs_1m"
     * @dataProvider invalidGs1mDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidGs1m(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setGs1m($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validGs1mDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_gs_1m'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidGs1mDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_gs_1m'));
    }

    /**
     * Test property "short"
     * @dataProvider validShortDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testShort(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getShort());
        self::assertEmpty($instance->short);
        $instance->setShort($value);
        self::assertEquals($value, is_array($value) ? $instance->getShort()->toArray() : $instance->getShort());
        self::assertEquals($value, is_array($value) ? $instance->short->toArray() : $instance->short);
        if (!empty($value)) {
            self::assertNotNull($instance->getShort());
            self::assertNotNull($instance->short);
            self::assertLessThanOrEqual(38, is_string($instance->getShort()) ? mb_strlen($instance->getShort()) : $instance->getShort());
            self::assertLessThanOrEqual(38, is_string($instance->short) ? mb_strlen($instance->short) : $instance->short);
            self::assertGreaterThanOrEqual(1, is_string($instance->getShort()) ? mb_strlen($instance->getShort()) : $instance->getShort());
            self::assertGreaterThanOrEqual(1, is_string($instance->short) ? mb_strlen($instance->short) : $instance->short);
        }
    }

    /**
     * Test invalid property "short"
     * @dataProvider invalidShortDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidShort(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setShort($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validShortDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_short'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidShortDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_short'));
    }

    /**
     * Test property "fur"
     * @dataProvider validFurDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testFur(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getFur());
        self::assertEmpty($instance->fur);
        $instance->setFur($value);
        self::assertEquals($value, is_array($value) ? $instance->getFur()->toArray() : $instance->getFur());
        self::assertEquals($value, is_array($value) ? $instance->fur->toArray() : $instance->fur);
        if (!empty($value)) {
            self::assertNotNull($instance->getFur());
            self::assertNotNull($instance->fur);
            self::assertLessThanOrEqual(20, is_string($instance->getFur()) ? mb_strlen($instance->getFur()) : $instance->getFur());
            self::assertLessThanOrEqual(20, is_string($instance->fur) ? mb_strlen($instance->fur) : $instance->fur);
            self::assertGreaterThanOrEqual(20, is_string($instance->getFur()) ? mb_strlen($instance->getFur()) : $instance->getFur());
            self::assertGreaterThanOrEqual(20, is_string($instance->fur) ? mb_strlen($instance->fur) : $instance->fur);
        }
    }

    /**
     * Test invalid property "fur"
     * @dataProvider invalidFurDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidFur(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setFur($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validFurDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_fur'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidFurDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_fur'));
    }

    /**
     * Test property "egais_20"
     * @dataProvider validEgais20DataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testEgais20(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getEgais20());
        self::assertEmpty($instance->egais_20);
        $instance->setEgais20($value);
        self::assertEquals($value, is_array($value) ? $instance->getEgais20()->toArray() : $instance->getEgais20());
        self::assertEquals($value, is_array($value) ? $instance->egais_20->toArray() : $instance->egais_20);
        if (!empty($value)) {
            self::assertNotNull($instance->getEgais20());
            self::assertNotNull($instance->egais_20);
            self::assertLessThanOrEqual(33, is_string($instance->getEgais20()) ? mb_strlen($instance->getEgais20()) : $instance->getEgais20());
            self::assertLessThanOrEqual(33, is_string($instance->egais_20) ? mb_strlen($instance->egais_20) : $instance->egais_20);
            self::assertGreaterThanOrEqual(33, is_string($instance->getEgais20()) ? mb_strlen($instance->getEgais20()) : $instance->getEgais20());
            self::assertGreaterThanOrEqual(33, is_string($instance->egais_20) ? mb_strlen($instance->egais_20) : $instance->egais_20);
        }
    }

    /**
     * Test invalid property "egais_20"
     * @dataProvider invalidEgais20DataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidEgais20(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setEgais20($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validEgais20DataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_egais_20'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidEgais20DataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_egais_20'));
    }

    /**
     * Test property "egais_30"
     * @dataProvider validEgais30DataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testEgais30(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getEgais30());
        self::assertEmpty($instance->egais_30);
        $instance->setEgais30($value);
        self::assertEquals($value, is_array($value) ? $instance->getEgais30()->toArray() : $instance->getEgais30());
        self::assertEquals($value, is_array($value) ? $instance->egais_30->toArray() : $instance->egais_30);
        if (!empty($value)) {
            self::assertNotNull($instance->getEgais30());
            self::assertNotNull($instance->egais_30);
            self::assertLessThanOrEqual(14, is_string($instance->getEgais30()) ? mb_strlen($instance->getEgais30()) : $instance->getEgais30());
            self::assertLessThanOrEqual(14, is_string($instance->egais_30) ? mb_strlen($instance->egais_30) : $instance->egais_30);
            self::assertGreaterThanOrEqual(14, is_string($instance->getEgais30()) ? mb_strlen($instance->getEgais30()) : $instance->getEgais30());
            self::assertGreaterThanOrEqual(14, is_string($instance->egais_30) ? mb_strlen($instance->egais_30) : $instance->egais_30);
        }
    }

    /**
     * Test invalid property "egais_30"
     * @dataProvider invalidEgais30DataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidEgais30(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setEgais30($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validEgais30DataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_egais_30'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidEgais30DataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_egais_30'));
    }

    /**
     * Test valid method "jsonSerialize"
     * @dataProvider validClassDataProvider
     * @param mixed $value
     *
     * @return void
     */
    public function testJsonSerialize(mixed $value): void
    {
        $instance = $this->getTestInstance($value);
        self::assertEquals($value, $instance->jsonSerialize());
    }

    /**
     * @return array
     * @throws Exception
     */
    public function validClassDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return [$this->getValidDataProviderByClass($instance)];
    }
}
