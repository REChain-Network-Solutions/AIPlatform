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

namespace Tests\YooKassa\Model\Payment\Confirmation;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payment\Confirmation\ConfirmationMobileApplication;

/**
 * ConfirmationMobileApplicationTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class ConfirmationMobileApplicationTest extends AbstractTestCase
{
    protected ConfirmationMobileApplication $object;

    /**
     * @return ConfirmationMobileApplication
     */
    protected function getTestInstance(): ConfirmationMobileApplication
    {
        return new ConfirmationMobileApplication();
    }

    /**
     * @return void
     */
    public function testConfirmationMobileApplicationClassExists(): void
    {
        $this->object = $this->getMockBuilder(ConfirmationMobileApplication::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(ConfirmationMobileApplication::class));
        $this->assertInstanceOf(ConfirmationMobileApplication::class, $this->object);
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
        self::assertContains($instance->getType(), ['mobile_application']);
        self::assertContains($instance->type, ['mobile_application']);
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
}
