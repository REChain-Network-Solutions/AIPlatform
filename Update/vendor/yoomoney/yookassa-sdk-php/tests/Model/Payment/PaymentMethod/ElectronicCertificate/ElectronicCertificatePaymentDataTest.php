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

namespace Tests\YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificatePaymentData;

/**
 * ElectronicCertificatePaymentDataTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
*/
class ElectronicCertificatePaymentDataTest extends AbstractTestCase
{
    protected ElectronicCertificatePaymentData $object;

    /**
    * @return ElectronicCertificatePaymentData
    */
    protected function getTestInstance(): ElectronicCertificatePaymentData
    {
        return new ElectronicCertificatePaymentData();
    }

    /**
    * @return void
    */
    public function testElectronicCertificatePaymentDataClassExists(): void
    {
        $this->object = $this->getMockBuilder(ElectronicCertificatePaymentData::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(ElectronicCertificatePaymentData::class));
        $this->assertInstanceOf(ElectronicCertificatePaymentData::class, $this->object);
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
        $instance->setAmount($value);
        self::assertNotNull($instance->getAmount());
        self::assertNotNull($instance->amount);
        self::assertEquals($value, is_array($value) ? $instance->getAmount()->toArray() : $instance->getAmount());
        self::assertEquals($value, is_array($value) ? $instance->amount->toArray() : $instance->amount);
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

    /**
    * Test property "basket_id"
    * @dataProvider validBasketIdDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testBasketId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setBasketId($value);
        self::assertNotNull($instance->getBasketId());
        self::assertNotNull($instance->basket_id);
        self::assertEquals($value, $instance->getBasketId());
        self::assertEquals($value, $instance->basket_id);
    }

    /**
    * Test invalid property "basket_id"
    * @dataProvider invalidBasketIdDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidBasketId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setBasketId($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validBasketIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_basket_id'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidBasketIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_basket_id'));
    }
}
