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

namespace Tests\YooKassa\Request\PaymentMethods;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use YooKassa\Helpers\Random;
use YooKassa\Request\PaymentMethods\PaymentMethodCard;


/**
 * CardRequestDataTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class PaymentMethodCardTest extends AbstractTestCase
{
    protected PaymentMethodCard $object;

    /**
     * @return PaymentMethodCard
     */
    protected function getTestInstance(): PaymentMethodCard
    {
        return new PaymentMethodCard();
    }

    /**
     * @return void
     */
    public function testPaymentMethodCardClassExists(): void
    {
        $this->object = $this->getMockBuilder(PaymentMethodCard::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(PaymentMethodCard::class));
        $this->assertInstanceOf(PaymentMethodCard::class, $this->object);
    }

    /**
     * Test property "number"
     * @dataProvider validNumberDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testNumber(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setNumber($value);
        self::assertNotNull($instance->getNumber());
        self::assertNotNull($instance->number);
        self::assertEquals($value, is_array($value) ? $instance->getNumber()->toArray() : $instance->getNumber());
        self::assertEquals($value, is_array($value) ? $instance->number->toArray() : $instance->number);
        self::assertMatchesRegularExpression("/[0-9]{14,19}/", $instance->getNumber());
        self::assertMatchesRegularExpression("/[0-9]{14,19}/", $instance->number);
    }

    /**
     * Test invalid property "number"
     * @dataProvider invalidNumberDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidNumber(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setNumber($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_number'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_number'));
    }

    /**
     * Test property "expiry_year"
     * @dataProvider validExpiryYearDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testExpiryYear(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setExpiryYear($value);
        self::assertNotNull($instance->getExpiryYear());
        self::assertNotNull($instance->expiry_year);
        self::assertEquals($value, is_array($value) ? $instance->getExpiryYear()->toArray() : $instance->getExpiryYear());
        self::assertEquals($value, is_array($value) ? $instance->expiry_year->toArray() : $instance->expiry_year);
        self::assertMatchesRegularExpression("/[0-9]{4}/", $instance->getExpiryYear());
        self::assertMatchesRegularExpression("/[0-9]{4}/", $instance->expiry_year);
    }

    /**
     * Test invalid property "expiry_year"
     * @dataProvider invalidExpiryYearDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidExpiryYear(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setExpiryYear($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validExpiryYearDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_expiry_year'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidExpiryYearDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_expiry_year'));
    }

    /**
     * Test property "expiry_month"
     * @dataProvider validExpiryMonthDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testExpiryMonth(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setExpiryMonth($value);
        self::assertNotNull($instance->getExpiryMonth());
        self::assertNotNull($instance->expiry_month);
        self::assertEquals($value, is_array($value) ? $instance->getExpiryMonth()->toArray() : $instance->getExpiryMonth());
        self::assertEquals($value, is_array($value) ? $instance->expiry_month->toArray() : $instance->expiry_month);
        self::assertMatchesRegularExpression("/^(0?[1-9]|1[0-2])$/", $instance->getExpiryMonth());
        self::assertMatchesRegularExpression("/^(0?[1-9]|1[0-2])$/", $instance->expiry_month);
    }

    /**
     * Test invalid property "expiry_month"
     * @dataProvider invalidExpiryMonthDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidExpiryMonth(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setExpiryMonth($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validExpiryMonthDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_expiry_month'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidExpiryMonthDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_expiry_month'));
    }

    /**
     * Test property "csc"
     * @dataProvider validCscDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCsc(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCsc());
        self::assertEmpty($instance->csc);
        $instance->setCsc($value);
        self::assertEquals($value, is_array($value) ? $instance->getCsc()->toArray() : $instance->getCsc());
        self::assertEquals($value, is_array($value) ? $instance->csc->toArray() : $instance->csc);
        if (!empty($value)) {
            self::assertNotNull($instance->getCsc());
            self::assertNotNull($instance->csc);
            self::assertMatchesRegularExpression("/[0-9]{3,4}/", $instance->getCsc());
            self::assertMatchesRegularExpression("/[0-9]{3,4}/", $instance->csc);
        }
    }

    /**
     * Test invalid property "csc"
     * @dataProvider invalidCscDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCsc(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCsc($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCscDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_csc'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCscDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_csc'));
    }

    /**
     * Test property "cardholder"
     * @dataProvider validCardholderDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCardholder(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCardholder());
        self::assertEmpty($instance->cardholder);
        $instance->setCardholder($value);
        self::assertEquals($value, is_array($value) ? $instance->getCardholder()->toArray() : $instance->getCardholder());
        self::assertEquals($value, is_array($value) ? $instance->cardholder->toArray() : $instance->cardholder);
        if (!empty($value)) {
            self::assertNotNull($instance->getCardholder());
            self::assertNotNull($instance->cardholder);
            self::assertMatchesRegularExpression("/[a-zA-Z '-]{1,26}/", $instance->getCardholder());
            self::assertMatchesRegularExpression("/[a-zA-Z '-]{1,26}/", $instance->cardholder);
        }
    }

    /**
     * Test invalid property "cardholder"
     * @dataProvider invalidCardholderDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCardholder(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCardholder($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCardholderDataProvider(): array
    {
        return [
            [null],
            [Random::str(1, 26, 'abcdefghijklmnopqrstuvwxyz \'-')],
            [Random::str(1, 26, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ \'-')],
        ];
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCardholderDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_cardholder'));
    }
}
