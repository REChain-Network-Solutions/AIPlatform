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

namespace Tests\YooKassa\Request\Payouts;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Request\Payouts\IncomeReceiptData;

/**
 * IncomeReceiptDataTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class IncomeReceiptDataTest extends AbstractTestCase
{
    protected IncomeReceiptData $object;

    /**
     * @return IncomeReceiptData
     */
    protected function getTestInstance(): IncomeReceiptData
    {
        return new IncomeReceiptData();
    }

    /**
     * @return void
     */
    public function testIncomeReceiptDataClassExists(): void
    {
        $this->object = $this->getMockBuilder(IncomeReceiptData::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(IncomeReceiptData::class));
        $this->assertInstanceOf(IncomeReceiptData::class, $this->object);
    }

    /**
     * Test property "service_name"
     * @dataProvider validServiceNameDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testServiceName(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setServiceName($value);
        self::assertNotNull($instance->getServiceName());
        self::assertNotNull($instance->service_name);
        self::assertEquals($value, is_array($value) ? $instance->getServiceName()->toArray() : $instance->getServiceName());
        self::assertEquals($value, is_array($value) ? $instance->service_name->toArray() : $instance->service_name);
        self::assertLessThanOrEqual(50, is_string($instance->getServiceName()) ? mb_strlen($instance->getServiceName()) : $instance->getServiceName());
        self::assertLessThanOrEqual(50, is_string($instance->service_name) ? mb_strlen($instance->service_name) : $instance->service_name);
        self::assertGreaterThanOrEqual(1, is_string($instance->getServiceName()) ? mb_strlen($instance->getServiceName()) : $instance->getServiceName());
        self::assertGreaterThanOrEqual(1, is_string($instance->service_name) ? mb_strlen($instance->service_name) : $instance->service_name);
    }

    /**
     * Test invalid property "service_name"
     * @dataProvider invalidServiceNameDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidServiceName(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setServiceName($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validServiceNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_service_name'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidServiceNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_service_name'));
    }

    /**
     * Test property "amount"
     * @dataProvider validAmountDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testAmount(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getAmount());
        self::assertEmpty($instance->amount);
        $instance->setAmount($value);
        self::assertEquals($value, is_array($value) ? $instance->getAmount()->toArray() : $instance->getAmount());
        self::assertEquals($value, is_array($value) ? $instance->amount->toArray() : $instance->amount);
        if (!empty($value)) {
            self::assertNotNull($instance->getAmount());
            self::assertNotNull($instance->amount);
        }
    }

    /**
     * Test invalid property "amount"
     * @dataProvider invalidAmountDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidAmount(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setAmount($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validAmountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_amount'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidAmountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_amount'));
    }
}
