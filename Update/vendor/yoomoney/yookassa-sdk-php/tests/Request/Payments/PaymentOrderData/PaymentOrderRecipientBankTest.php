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
use YooKassa\Request\Payments\PaymentOrderData\PaymentOrderRecipientBank;

/**
 * PaymentOrderRecipientBankTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
*/
class PaymentOrderRecipientBankTest extends AbstractTestCase
{
    protected PaymentOrderRecipientBank $object;

    /**
    * @return PaymentOrderRecipientBank
    */
    protected function getTestInstance(): PaymentOrderRecipientBank
    {
        return new PaymentOrderRecipientBank();
    }

    /**
    * @return void
    */
    public function testPaymentOrderRecipientBankClassExists(): void
    {
        $this->object = $this->getMockBuilder(PaymentOrderRecipientBank::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(PaymentOrderRecipientBank::class));
        $this->assertInstanceOf(PaymentOrderRecipientBank::class, $this->object);
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
    * Test property "bic"
    * @dataProvider validBicDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testBic(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setBic($value);
        self::assertNotNull($instance->getBic());
        self::assertNotNull($instance->bic);
        self::assertEquals($value, is_array($value) ? $instance->getBic()->toArray() : $instance->getBic());
        self::assertEquals($value, is_array($value) ? $instance->bic->toArray() : $instance->bic);
        self::assertMatchesRegularExpression("/[0-9]{9}/", $instance->getBic());
        self::assertMatchesRegularExpression("/[0-9]{9}/", $instance->bic);
    }

    /**
    * Test invalid property "bic"
    * @dataProvider invalidBicDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidBic(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setBic($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validBicDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_bic'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidBicDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_bic'));
    }

    /**
    * Test property "account"
    * @dataProvider validAccountDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testAccount(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setAccount($value);
        self::assertNotNull($instance->getAccount());
        self::assertNotNull($instance->account);
        self::assertEquals($value, is_array($value) ? $instance->getAccount()->toArray() : $instance->getAccount());
        self::assertEquals($value, is_array($value) ? $instance->account->toArray() : $instance->account);
    }

    /**
    * Test invalid property "account"
    * @dataProvider invalidAccountDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidAccount(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setAccount($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validAccountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_account'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidAccountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_account'));
    }

    /**
    * Test property "correspondent_account"
    * @dataProvider validCorrespondentAccountDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testCorrespondentAccount(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setCorrespondentAccount($value);
        self::assertNotNull($instance->getCorrespondentAccount());
        self::assertNotNull($instance->correspondent_account);
        self::assertEquals($value, is_array($value) ? $instance->getCorrespondentAccount()->toArray() : $instance->getCorrespondentAccount());
        self::assertEquals($value, is_array($value) ? $instance->correspondent_account->toArray() : $instance->correspondent_account);
    }

    /**
    * Test invalid property "correspondent_account"
    * @dataProvider invalidCorrespondentAccountDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidCorrespondentAccount(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCorrespondentAccount($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validCorrespondentAccountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_correspondent_account'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidCorrespondentAccountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_correspondent_account'));
    }
}
