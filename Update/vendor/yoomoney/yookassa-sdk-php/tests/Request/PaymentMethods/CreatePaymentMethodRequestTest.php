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
use YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest;

/**
 * CreatePaymentMethodRequestTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
*/
class CreatePaymentMethodRequestTest extends AbstractTestCase
{
    protected CreatePaymentMethodRequest $object;

    /**
    * @return CreatePaymentMethodRequest
    */
    protected function getTestInstance(): CreatePaymentMethodRequest
    {
        return new CreatePaymentMethodRequest();
    }

    /**
    * @return void
    */
    public function testCreatePaymentMethodRequestClassExists(): void
    {
        $this->object = $this->getMockBuilder(CreatePaymentMethodRequest::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(CreatePaymentMethodRequest::class));
        $this->assertInstanceOf(CreatePaymentMethodRequest::class, $this->object);
    }

    /**
    * Test property "type"
    * @dataProvider validTypeDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testType(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertFalse($instance->hasType());
        $instance->setType($value);
        self::assertNotNull($instance->getType());
        self::assertNotNull($instance->type);
        self::assertTrue($instance->hasType());
        self::assertEquals($value, is_array($value) ? $instance->getType()->toArray() : $instance->getType());
        self::assertEquals($value, is_array($value) ? $instance->type->toArray() : $instance->type);
        self::assertContains($instance->getType(), ['bank_card']);
        self::assertContains($instance->type, ['bank_card']);
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
    public function validTypeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_type'));
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

    /**
    * Test property "holder"
    * @dataProvider validHolderDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testHolder(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getHolder());
        self::assertEmpty($instance->holder);
        $instance->setHolder($value);
        self::assertEquals($value, is_array($value) ? $instance->getHolder()->toArray() : $instance->getHolder());
        self::assertEquals($value, is_array($value) ? $instance->holder->toArray() : $instance->holder);
        if (!empty($value)) {
            self::assertNotNull($instance->getHolder());
            self::assertNotNull($instance->holder);
        }
    }

    /**
    * Test invalid property "holder"
    * @dataProvider invalidHolderDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidHolder(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setHolder($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validHolderDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_holder'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidHolderDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_holder'));
    }

    /**
    * Test property "client_ip"
    * @dataProvider validClientIpDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testClientIp(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getClientIp());
        self::assertEmpty($instance->client_ip);
        $instance->setClientIp($value);
        self::assertEquals($value, is_array($value) ? $instance->getClientIp()->toArray() : $instance->getClientIp());
        self::assertEquals($value, is_array($value) ? $instance->client_ip->toArray() : $instance->client_ip);
        if (!empty($value)) {
            self::assertNotNull($instance->getClientIp());
            self::assertNotNull($instance->client_ip);
        }
    }

    /**
    * Test invalid property "client_ip"
    * @dataProvider invalidClientIpDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidClientIp(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setClientIp($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validClientIpDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_client_ip'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidClientIpDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_client_ip'));
    }

    /**
    * Test property "confirmation"
    * @dataProvider validConfirmationDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testConfirmation(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getConfirmation());
        self::assertEmpty($instance->confirmation);
        $instance->setConfirmation($value);
        self::assertEquals($value, is_array($value) ? $instance->getConfirmation()->toArray() : $instance->getConfirmation());
        self::assertEquals($value, is_array($value) ? $instance->confirmation->toArray() : $instance->confirmation);
        if (!empty($value)) {
            self::assertNotNull($instance->getConfirmation());
            self::assertNotNull($instance->confirmation);
        }
    }

    /**
    * Test invalid property "confirmation"
    * @dataProvider invalidConfirmationDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidConfirmation(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setConfirmation($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validConfirmationDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_confirmation'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidConfirmationDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_confirmation'));
    }
}
