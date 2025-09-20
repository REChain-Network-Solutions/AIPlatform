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

namespace Tests\YooKassa\Model\Deal;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Deal\RefundDealData;
use YooKassa\Model\Metadata;

/**
 * RefundDealDataTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class RefundDealDataTest extends AbstractTestCase
{
    protected RefundDealData $object;

    /**
     * @return RefundDealData
     */
    protected function getTestInstance(): RefundDealData
    {
        return new RefundDealData();
    }

    /**
     * @return void
     */
    public function testRefundDealDataClassExists(): void
    {
        $this->object = $this->getMockBuilder(RefundDealData::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(RefundDealData::class));
        $this->assertInstanceOf(RefundDealData::class, $this->object);
    }

    /**
     * Test property "refund_settlements"
     * @dataProvider validRefundSettlementsDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testRefundSettlements(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertIsObject($instance->getRefundSettlements());
        self::assertIsObject($instance->refund_settlements);
        self::assertCount(0, $instance->getRefundSettlements());
        self::assertCount(0, $instance->refund_settlements);
        $instance->setRefundSettlements($value);
        self::assertNotNull($instance->getRefundSettlements());
        self::assertNotNull($instance->refund_settlements);
        foreach ($value as $key => $element) {
            if (is_array($element) && !empty($element)) {
                self::assertEquals($element, $instance->getRefundSettlements()[$key]->toArray());
                self::assertEquals($element, $instance->refund_settlements[$key]->toArray());
                self::assertIsArray($instance->getRefundSettlements()[$key]->toArray());
                self::assertIsArray($instance->refund_settlements[$key]->toArray());
            }
            if (is_object($element) && !empty($element)) {
                self::assertEquals($element, $instance->getRefundSettlements()->get($key));
                self::assertIsObject($instance->getRefundSettlements()->get($key));
                self::assertIsObject($instance->refund_settlements->get($key));
                self::assertIsObject($instance->getRefundSettlements());
                self::assertIsObject($instance->refund_settlements);
            }
        }
        self::assertCount(count($value), $instance->getRefundSettlements());
        self::assertCount(count($value), $instance->refund_settlements);
    }

    /**
     * Test invalid property "refund_settlements"
     * @dataProvider invalidRefundSettlementsDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidRefundSettlements(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setRefundSettlements($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validRefundSettlementsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_refund_settlements'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidRefundSettlementsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_refund_settlements'));
    }
}
