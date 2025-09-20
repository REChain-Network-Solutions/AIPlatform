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
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payment\PaymentMethod\PaymentMethodSberLoan;

/**
 * PaymentMethodSberLoanTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class PaymentMethodSberLoanTest extends AbstractTestCase
{
    protected PaymentMethodSberLoan $object;

    /**
     * @return PaymentMethodSberLoan
     */
    protected function getTestInstance(): PaymentMethodSberLoan
    {
        return new PaymentMethodSberLoan();
    }

    /**
     * @return void
     */
    public function testPaymentMethodSberLoanClassExists(): void
    {
        $this->object = $this->getMockBuilder(PaymentMethodSberLoan::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(PaymentMethodSberLoan::class));
        $this->assertInstanceOf(PaymentMethodSberLoan::class, $this->object);
    }

    /**
     * Test property "type"
     *
     * @return void
     * @throws Exception
     */
    public function testType(): void
    {
        $instance = $this->getTestInstance();
        self::assertContains($instance->getType(), ['sber_loan']);
        self::assertContains($instance->type, ['sber_loan']);
        self::assertNotNull($instance->getType());
        self::assertNotNull($instance->type);
    }

    /**
     * Test invalid property "type"
     * @dataProvider invalidTypeDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidType(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setType($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidTypeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_type'));
    }

    /**
     * Test property "loan_option"
     * @dataProvider validLoanOptionDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testLoanOption(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getLoanOption());
        self::assertEmpty($instance->loan_option);
        $instance->setLoanOption($value);
        self::assertEquals($value, is_array($value) ? $instance->getLoanOption()->toArray() : $instance->getLoanOption());
        self::assertEquals($value, is_array($value) ? $instance->loan_option->toArray() : $instance->loan_option);
        if (!empty($value)) {
            self::assertNotNull($instance->getLoanOption());
            self::assertNotNull($instance->loan_option);
            self::assertGreaterThanOrEqual(1, is_string($instance->getLoanOption()) ? mb_strlen($instance->getLoanOption()) : $instance->getLoanOption());
            self::assertGreaterThanOrEqual(1, is_string($instance->loan_option) ? mb_strlen($instance->loan_option) : $instance->loan_option);
        }
    }

    /**
     * Test invalid property "loan_option"
     * @dataProvider invalidLoanOptionDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidLoanOption(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setLoanOption($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validLoanOptionDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_loan_option'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidLoanOptionDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_loan_option'));
    }

    /**
     * Test property "discount_amount"
     * @dataProvider validDiscountAmountDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testDiscountAmount(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getDiscountAmount());
        self::assertEmpty($instance->discount_amount);
        $instance->setDiscountAmount($value);
        self::assertEquals($value, is_array($value) ? $instance->getDiscountAmount()->toArray() : $instance->getDiscountAmount());
        self::assertEquals($value, is_array($value) ? $instance->discount_amount->toArray() : $instance->discount_amount);
        if (!empty($value)) {
            self::assertNotNull($instance->getDiscountAmount());
            self::assertNotNull($instance->discount_amount);
        }
    }

    /**
     * Test invalid property "discount_amount"
     * @dataProvider invalidDiscountAmountDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidDiscountAmount(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setDiscountAmount($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validDiscountAmountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_discount_amount'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidDiscountAmountDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_discount_amount'));
    }

    /**
     * Test property "suspended_until"
     * @dataProvider validSuspendedUntilDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testSuspendedUntil(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getSuspendedUntil());
        self::assertEmpty($instance->suspended_until);
        $instance->setSuspendedUntil($value);
        if (!empty($value)) {
            self::assertNotNull($instance->getSuspendedUntil());
            self::assertNotNull($instance->suspended_until);
            if ($value instanceof Datetime) {
                self::assertEquals($value, $instance->getSuspendedUntil());
                self::assertEquals($value, $instance->suspended_until);
            } else {
                self::assertEquals(new Datetime($value), $instance->getSuspendedUntil());
                self::assertEquals(new Datetime($value), $instance->suspended_until);
            }
        }
    }

    /**
     * Test invalid property "suspended_until"
     * @dataProvider invalidSuspendedUntilDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidSuspendedUntil(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setSuspendedUntil($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validSuspendedUntilDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_suspended_until'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidSuspendedUntilDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_suspended_until'));
    }
}
