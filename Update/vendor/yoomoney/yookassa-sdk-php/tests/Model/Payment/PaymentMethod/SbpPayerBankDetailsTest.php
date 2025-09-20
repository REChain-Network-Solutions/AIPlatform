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

namespace Tests\YooKassa\Model\Payment\PaymentMethod;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use YooKassa\Model\Payment\PaymentMethod\SbpPayerBankDetails;

/**
 * SbpPayerBankDetailsTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
*/
class SbpPayerBankDetailsTest extends AbstractTestCase
{
    protected SbpPayerBankDetails $object;

    /**
    * @return SbpPayerBankDetails
    */
    protected function getTestInstance(): SbpPayerBankDetails
    {
        return new SbpPayerBankDetails();
    }

    /**
    * @return void
    */
    public function testSbpPayerBankDetailsClassExists(): void
    {
        $this->object = $this->getMockBuilder(SbpPayerBankDetails::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(SbpPayerBankDetails::class));
        $this->assertInstanceOf(SbpPayerBankDetails::class, $this->object);
    }

    /**
    * Test property "bank_id"
    * @dataProvider validBankIdDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testBankId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setBankId($value);
        self::assertNotNull($instance->getBankId());
        self::assertNotNull($instance->bank_id);
        self::assertEquals($value, $instance->getBankId());
        self::assertEquals($value, $instance->bank_id);
        self::assertMatchesRegularExpression("/[a-zA-Z0-9]{12}/", $instance->getBankId());
        self::assertMatchesRegularExpression("/[a-zA-Z0-9]{12}/", $instance->bank_id);
        self::assertEquals(12, is_string($instance->getBankId()) ? mb_strlen($instance->getBankId()) : $instance->getBankId());
        self::assertEquals(12, is_string($instance->bank_id) ? mb_strlen($instance->bank_id) : $instance->bank_id);
    }

    /**
    * Test invalid property "bank_id"
    * @dataProvider invalidBankIdDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidBankId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setBankId($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validBankIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_bank_id'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidBankIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_bank_id'));
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
        self::assertEquals($value, $instance->getBic());
        self::assertEquals($value, $instance->bic);
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
}
