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

namespace Tests\YooKassa\Model\SavePaymentMethod\Confirmation;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payment\Confirmation\ConfirmationRedirect;

/**
 * ConfirmationRedirectTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
*/
class ConfirmationRedirectTest extends AbstractTestCase
{
    protected ConfirmationRedirect $object;

    /**
    * @return ConfirmationRedirect
    */
    protected function getTestInstance(): ConfirmationRedirect
    {
        return new ConfirmationRedirect();
    }

    /**
    * @return void
    */
    public function testConfirmationRedirectClassExists(): void
    {
        $this->object = $this->getMockBuilder(ConfirmationRedirect::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(ConfirmationRedirect::class));
        $this->assertInstanceOf(ConfirmationRedirect::class, $this->object);
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
        $instance->setType($value);
        self::assertNotNull($instance->getType());
        self::assertNotNull($instance->type);
        self::assertEquals($value, is_array($value) ? $instance->getType()->toArray() : $instance->getType());
        self::assertEquals($value, is_array($value) ? $instance->type->toArray() : $instance->type);
        self::assertContains($instance->getType(), ['redirect']);
        self::assertContains($instance->type, ['redirect']);
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
        return [['redirect']];
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
    * Test property "confirmation_url"
    * @dataProvider validConfirmationUrlDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testConfirmationUrl(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setConfirmationUrl($value);
        self::assertNotNull($instance->getConfirmationUrl());
        self::assertNotNull($instance->confirmation_url);
        self::assertEquals($value, is_array($value) ? $instance->getConfirmationUrl()->toArray() : $instance->getConfirmationUrl());
        self::assertEquals($value, is_array($value) ? $instance->confirmation_url->toArray() : $instance->confirmation_url);
    }

    /**
    * Test invalid property "confirmation_url"
    * @dataProvider invalidConfirmationUrlDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidConfirmationUrl(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setConfirmationUrl($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validConfirmationUrlDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_confirmation_url'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidConfirmationUrlDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_confirmation_url'));
    }

    /**
    * Test property "enforce"
    * @dataProvider validEnforceDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testEnforce(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getEnforce());
        self::assertEmpty($instance->enforce);
        $instance->setEnforce($value);
        self::assertEquals($value, is_array($value) ? $instance->getEnforce()->toArray() : $instance->getEnforce());
        self::assertEquals($value, is_array($value) ? $instance->enforce->toArray() : $instance->enforce);
        if (!empty($value)) {
            self::assertNotNull($instance->getEnforce());
            self::assertNotNull($instance->enforce);
        }
    }

    /**
    * Test invalid property "enforce"
    * @dataProvider invalidEnforceDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidEnforce(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setEnforce($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validEnforceDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_enforce'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidEnforceDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_enforce'));
    }

    /**
    * Test property "return_url"
    * @dataProvider validReturnUrlDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testReturnUrl(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getReturnUrl());
        self::assertEmpty($instance->return_url);
        $instance->setReturnUrl($value);
        self::assertEquals($value, is_array($value) ? $instance->getReturnUrl()->toArray() : $instance->getReturnUrl());
        self::assertEquals($value, is_array($value) ? $instance->return_url->toArray() : $instance->return_url);
        if (!empty($value)) {
            self::assertNotNull($instance->getReturnUrl());
            self::assertNotNull($instance->return_url);
        }
    }

    /**
    * Test invalid property "return_url"
    * @dataProvider invalidReturnUrlDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidReturnUrl(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setReturnUrl($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validReturnUrlDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_return_url'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidReturnUrlDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_return_url'));
    }
}
