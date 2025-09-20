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

namespace Tests\YooKassa\Request\Payments\PaymentOrderData;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipient;

/**
 * PaymentOrderRecipientTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
*/
class PaymentOrderRecipientTest extends AbstractTestCase
{
    protected PaymentOrderRecipient $object;

    /**
    * @return PaymentOrderRecipient
    */
    protected function getTestInstance(): PaymentOrderRecipient
    {
        return new PaymentOrderRecipient();
    }

    /**
    * @return void
    */
    public function testPaymentOrderRecipientClassExists(): void
    {
        $this->object = $this->getMockBuilder(PaymentOrderRecipient::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(PaymentOrderRecipient::class));
        $this->assertInstanceOf(PaymentOrderRecipient::class, $this->object);
    }

    /**
    * Test property "name"
    * @dataProvider validNameDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testName(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setName($value);
        self::assertNotNull($instance->getName());
        self::assertNotNull($instance->name);
        self::assertEquals($value, is_array($value) ? $instance->getName()->toArray() : $instance->getName());
        self::assertEquals($value, is_array($value) ? $instance->name->toArray() : $instance->name);
    }

    /**
    * Test invalid property "name"
    * @dataProvider invalidNameDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidName(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setName($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_name'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidNameDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_name'));
    }

    /**
    * Test property "inn"
    * @dataProvider validInnDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testInn(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setInn($value);
        self::assertNotNull($instance->getInn());
        self::assertNotNull($instance->inn);
        self::assertEquals($value, is_array($value) ? $instance->getInn()->toArray() : $instance->getInn());
        self::assertEquals($value, is_array($value) ? $instance->inn->toArray() : $instance->inn);
        self::assertMatchesRegularExpression("/[0-9]{10}/", $instance->getInn());
        self::assertMatchesRegularExpression("/[0-9]{10}/", $instance->inn);
    }

    /**
    * Test invalid property "inn"
    * @dataProvider invalidInnDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidInn(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setInn($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validInnDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_inn'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidInnDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_inn'));
    }

    /**
    * Test property "kpp"
    * @dataProvider validKppDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testKpp(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setKpp($value);
        self::assertNotNull($instance->getKpp());
        self::assertNotNull($instance->kpp);
        self::assertEquals($value, is_array($value) ? $instance->getKpp()->toArray() : $instance->getKpp());
        self::assertEquals($value, is_array($value) ? $instance->kpp->toArray() : $instance->kpp);
        self::assertMatchesRegularExpression("/[0-9]{9}/", $instance->getKpp());
        self::assertMatchesRegularExpression("/[0-9]{9}/", $instance->kpp);
    }

    /**
    * Test invalid property "kpp"
    * @dataProvider invalidKppDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidKpp(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setKpp($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validKppDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_kpp'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidKppDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_kpp'));
    }

    /**
    * Test property "bank"
    * @dataProvider validBankDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testBank(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setBank($value);
        self::assertNotNull($instance->getBank());
        self::assertNotNull($instance->bank);
        self::assertEquals($value, is_array($value) ? $instance->getBank()->toArray() : $instance->getBank());
        self::assertEquals($value, is_array($value) ? $instance->bank->toArray() : $instance->bank);
    }

    /**
    * Test invalid property "bank"
    * @dataProvider invalidBankDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidBank(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setBank($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validBankDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_bank'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidBankDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_bank'));
    }
}
