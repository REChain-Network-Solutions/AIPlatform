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
use YooKassa\Model\Payment\PaymentMethod\PaymentMethodSberbank;

/**
 * PaymentMethodSberbankTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class PaymentMethodSberbankTest extends AbstractTestCase
{
    protected PaymentMethodSberbank $object;

    /**
     * @return PaymentMethodSberbank
     */
    protected function getTestInstance(): PaymentMethodSberbank
    {
        return new PaymentMethodSberbank();
    }

    /**
     * @return void
     */
    public function testPaymentMethodSberbankClassExists(): void
    {
        $this->object = $this->getMockBuilder(PaymentMethodSberbank::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(PaymentMethodSberbank::class));
        $this->assertInstanceOf(PaymentMethodSberbank::class, $this->object);
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
        self::assertContains($instance->getType(), ['sberbank']);
        self::assertContains($instance->type, ['sberbank']);
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
     * Test property "phone"
     * @dataProvider validPhoneDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPhone(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setPhone($value);
        self::assertNotNull($instance->getPhone());
        self::assertNotNull($instance->phone);
        self::assertEquals($value, is_array($value) ? $instance->getPhone()->toArray() : $instance->getPhone());
        self::assertEquals($value, is_array($value) ? $instance->phone->toArray() : $instance->phone);
        self::assertMatchesRegularExpression("/[0-9]{4,15}/", $instance->getPhone());
        self::assertMatchesRegularExpression("/[0-9]{4,15}/", $instance->phone);
    }

    /**
     * Test invalid property "phone"
     * @dataProvider invalidPhoneDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPhone(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPhone($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPhoneDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_phone'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPhoneDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_phone'));
    }

    /**
     * Test property "card"
     * @dataProvider validCardDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCard(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCard());
        self::assertEmpty($instance->card);
        $instance->setCard($value);
        self::assertEquals($value, is_array($value) ? $instance->getCard()->toArray() : $instance->getCard());
        self::assertEquals($value, is_array($value) ? $instance->card->toArray() : $instance->card);
        if (!empty($value)) {
            self::assertNotNull($instance->getCard());
            self::assertNotNull($instance->card);
        }
    }

    /**
     * Test invalid property "card"
     * @dataProvider invalidCardDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCard(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCard($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCardDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_card'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCardDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_card'));
    }
}
