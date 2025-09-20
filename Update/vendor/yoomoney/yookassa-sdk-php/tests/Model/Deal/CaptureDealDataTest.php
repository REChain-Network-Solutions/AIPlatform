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
use YooKassa\Model\Metadata;
use YooKassa\Model\Deal\CaptureDealData;

/**
 * CaptureDealDataTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class CaptureDealDataTest extends AbstractTestCase
{
    protected CaptureDealData $object;

    /**
     * @return CaptureDealData
     */
    protected function getTestInstance(): CaptureDealData
    {
        return new CaptureDealData();
    }

    /**
     * @return void
     */
    public function testPaymentCaptureRequestDealClassExists(): void
    {
        $this->object = $this->getMockBuilder(CaptureDealData::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(CaptureDealData::class));
        $this->assertInstanceOf(CaptureDealData::class, $this->object);
    }

    /**
     * Test property "settlements"
     * @dataProvider validSettlementsDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testSettlements(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertIsObject($instance->getSettlements());
        self::assertIsObject($instance->settlements);
        self::assertCount(0, $instance->getSettlements());
        self::assertCount(0, $instance->settlements);
        $instance->setSettlements($value);
        self::assertNotNull($instance->getSettlements());
        self::assertNotNull($instance->settlements);
        foreach ($value as $key => $element) {
            if (is_array($element) && !empty($element)) {
                self::assertEquals($element, $instance->getSettlements()[$key]->toArray());
                self::assertEquals($element, $instance->settlements[$key]->toArray());
                self::assertIsArray($instance->getSettlements()[$key]->toArray());
                self::assertIsArray($instance->settlements[$key]->toArray());
            }
            if (is_object($element) && !empty($element)) {
                self::assertEquals($element, $instance->getSettlements()->get($key));
                self::assertIsObject($instance->getSettlements()->get($key));
                self::assertIsObject($instance->settlements->get($key));
                self::assertIsObject($instance->getSettlements());
                self::assertIsObject($instance->settlements);
            }
        }
        self::assertCount(count($value), $instance->getSettlements());
        self::assertCount(count($value), $instance->settlements);
    }

    /**
     * Test invalid property "settlements"
     * @dataProvider invalidSettlementsDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidSettlements(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setSettlements($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validSettlementsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_settlements'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidSettlementsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_settlements'));
    }
}
