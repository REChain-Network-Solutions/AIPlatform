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
use YooKassa\Request\Payments\FraudData;

/**
 * FraudDataTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class FraudDataTest extends AbstractTestCase
{
    protected FraudData $object;

    /**
     * @return FraudData
     */
    protected function getTestInstance(): FraudData
    {
        return new FraudData();
    }

    /**
     * @return void
     */
    public function testFraudDataClassExists(): void
    {
        $this->object = $this->getMockBuilder(FraudData::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(FraudData::class));
        $this->assertInstanceOf(FraudData::class, $this->object);
    }

    /**
     * Test property "topped_up_phone"
     * @dataProvider validToppedUpPhoneDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testToppedUpPhone(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getToppedUpPhone());
        self::assertEmpty($instance->topped_up_phone);
        $instance->setToppedUpPhone($value);
        self::assertEquals($value, is_array($value) ? $instance->getToppedUpPhone()->toArray() : $instance->getToppedUpPhone());
        self::assertEquals($value, is_array($value) ? $instance->topped_up_phone->toArray() : $instance->topped_up_phone);
        if (!empty($value)) {
            self::assertNotNull($instance->getToppedUpPhone());
            self::assertNotNull($instance->topped_up_phone);
            self::assertMatchesRegularExpression("/[0-9]{4,15}/", $instance->getToppedUpPhone());
            self::assertMatchesRegularExpression("/[0-9]{4,15}/", $instance->topped_up_phone);
        }
    }

    /**
     * Test invalid property "topped_up_phone"
     * @dataProvider invalidToppedUpPhoneDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidToppedUpPhone(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setToppedUpPhone($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validToppedUpPhoneDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_topped_up_phone'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidToppedUpPhoneDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_topped_up_phone'));
    }

    /**
     * Test property "merchant_customer_bank_account"
     * @dataProvider validMerchantCustomerBankAccountDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testMerchantCustomerBankAccount(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getMerchantCustomerBankAccount());
        self::assertEmpty($instance->merchant_customer_bank_account);
        $instance->setMerchantCustomerBankAccount($value);
        self::assertEquals($value, is_array($value) ? $instance->getMerchantCustomerBankAccount()->toArray() : $instance->getMerchantCustomerBankAccount());
        self::assertEquals($value, is_array($value) ? $instance->merchant_customer_bank_account->toArray() : $instance->merchant_customer_bank_account);
        if (!empty($value)) {
            self::assertNotNull($instance->getMerchantCustomerBankAccount());
            self::assertNotNull($instance->merchant_customer_bank_account);
        }
    }

    /**
     * Test invalid property "merchant_customer_bank_account"
     * @dataProvider invalidMerchantCustomerBankAccountDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidMerchantCustomerBankAccount(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setMerchantCustomerBankAccount($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validMerchantCustomerBankAccountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_merchant_customer_bank_account'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidMerchantCustomerBankAccountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_merchant_customer_bank_account'));
    }
}
