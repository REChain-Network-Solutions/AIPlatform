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

namespace Tests\YooKassa\Request\SelfEmployed;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Common\AbstractRequestBuilder;
use YooKassa\Model\Metadata;
use YooKassa\Request\SelfEmployed\SelfEmployedRequest;

/**
 * SelfEmployedRequestTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class SelfEmployedRequestTest extends AbstractTestCase
{
    protected SelfEmployedRequest $object;

    /**
     * @param mixed|null $value
     * @return SelfEmployedRequest
     */
    protected function getTestInstance(mixed $value = null): SelfEmployedRequest
    {
        return new SelfEmployedRequest($value);
    }

    /**
     * @return void
     */
    public function testSelfEmployedRequestClassExists(): void
    {
        $this->object = $this->getMockBuilder(SelfEmployedRequest::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(SelfEmployedRequest::class));
        $this->assertInstanceOf(SelfEmployedRequest::class, $this->object);
    }

    /**
     * Test property "itn"
     * @dataProvider validItnDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testItn(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getItn());
        self::assertEmpty($instance->itn);
        $instance->setItn($value);
        self::assertEquals($value, is_array($value) ? $instance->getItn()->toArray() : $instance->getItn());
        self::assertEquals($value, is_array($value) ? $instance->itn->toArray() : $instance->itn);
        if (!empty($value)) {
            self::assertTrue($instance->hasItn());
            self::assertNotNull($instance->getItn());
            self::assertNotNull($instance->itn);
            self::assertMatchesRegularExpression("/^\d{12}$/", $instance->getItn());
            self::assertMatchesRegularExpression("/^\d{12}$/", $instance->itn);
        }
    }

    /**
     * Test invalid property "itn"
     * @dataProvider invalidItnDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidItn(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setItn($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validItnDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_itn'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidItnDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_itn'));
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
        self::assertEmpty($instance->getPhone());
        self::assertEmpty($instance->phone);
        $instance->setPhone($value);
        self::assertEquals($value, is_array($value) ? $instance->getPhone()->toArray() : $instance->getPhone());
        self::assertEquals($value, is_array($value) ? $instance->phone->toArray() : $instance->phone);
        if (!empty($value)) {
            self::assertTrue($instance->hasPhone());
            self::assertNotNull($instance->getPhone());
            self::assertNotNull($instance->phone);
            self::assertMatchesRegularExpression("/[0-9]{4,15}/", $instance->getPhone());
            self::assertMatchesRegularExpression("/[0-9]{4,15}/", $instance->phone);
        }
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
            self::assertTrue($instance->hasConfirmation());
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

    public function testValidate(): void
    {
        $instance = new SelfEmployedRequest();
        self::assertFalse($instance->validate());
        $instance->setPhone(+79999999999);
        self::assertTrue($instance->validate());
    }

    /**
     * Test valid method "builder"
     * @dataProvider validClassDataProvider
     * @param mixed $value
     *
     * @return void
     */
    public function testBuilder(mixed $value): void
    {
        $instance = $this->getTestInstance($value);
        self::assertInstanceOf(AbstractRequestBuilder::class, $instance::builder());
    }

    /**
     * @return array
     * @throws Exception
     */
    public function validClassDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return [$this->getValidDataProviderByClass($instance)];
    }
}
