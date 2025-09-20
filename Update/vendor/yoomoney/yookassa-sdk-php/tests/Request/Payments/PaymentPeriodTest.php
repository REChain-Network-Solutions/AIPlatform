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

namespace Tests\YooKassa\Request\Payments;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Request\Payments\PaymentPeriod;

/**
 * PaymentPeriodTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
*/
class PaymentPeriodTest extends AbstractTestCase
{
    protected PaymentPeriod $object;

    /**
    * @return PaymentPeriod
    */
    protected function getTestInstance(): PaymentPeriod
    {
        return new PaymentPeriod();
    }

    /**
    * @return void
    */
    public function testPaymentPeriodClassExists(): void
    {
        $this->object = $this->getMockBuilder(PaymentPeriod::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(PaymentPeriod::class));
        $this->assertInstanceOf(PaymentPeriod::class, $this->object);
    }

    /**
    * Test property "month"
    * @dataProvider validMonthDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testMonth(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setMonth($value);
        self::assertNotNull($instance->getMonth());
        self::assertNotNull($instance->month);
        self::assertEquals($value, $instance->getMonth());
        self::assertEquals($value, $instance->month);
        self::assertLessThanOrEqual(12, is_string($instance->getMonth()) ? mb_strlen($instance->getMonth()) : $instance->getMonth());
        self::assertLessThanOrEqual(12, is_string($instance->month) ? mb_strlen($instance->month) : $instance->month);
        self::assertGreaterThanOrEqual(1, is_string($instance->getMonth()) ? mb_strlen($instance->getMonth()) : $instance->getMonth());
        self::assertGreaterThanOrEqual(1, is_string($instance->month) ? mb_strlen($instance->month) : $instance->month);
        self::assertIsNumeric($instance->getMonth());
        self::assertIsNumeric($instance->month);
    }

    /**
    * Test invalid property "month"
    * @dataProvider invalidMonthDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidMonth(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setMonth($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validMonthDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_month'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidMonthDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_month'));
    }

    /**
    * Test property "year"
    * @dataProvider validYearDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testYear(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setYear($value);
        self::assertNotNull($instance->getYear());
        self::assertNotNull($instance->year);
        self::assertEquals($value, $instance->getYear());
        self::assertEquals($value, $instance->year);
        self::assertIsNumeric($instance->getYear());
        self::assertIsNumeric($instance->year);
    }

    /**
    * Test invalid property "year"
    * @dataProvider invalidYearDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidYear(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setYear($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validYearDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_year'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidYearDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_year'));
    }
}
