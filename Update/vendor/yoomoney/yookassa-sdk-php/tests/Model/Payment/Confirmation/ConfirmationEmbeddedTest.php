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
use YooKassa\Model\Payment\Confirmation\ConfirmationEmbedded;

/**
 * ConfirmationEmbeddedTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class ConfirmationEmbeddedTest extends AbstractTestCase
{
    protected ConfirmationEmbedded $object;

    /**
     * @return ConfirmationEmbedded
     */
    protected function getTestInstance(): ConfirmationEmbedded
    {
        return new ConfirmationEmbedded();
    }

    /**
     * @return void
     */
    public function testConfirmationEmbeddedClassExists(): void
    {
        $this->object = $this->getMockBuilder(ConfirmationEmbedded::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(ConfirmationEmbedded::class));
        $this->assertInstanceOf(ConfirmationEmbedded::class, $this->object);
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
        self::assertContains($instance->getType(), ['embedded']);
        self::assertContains($instance->type, ['embedded']);
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
     * Test property "confirmation_token"
     * @dataProvider validConfirmationTokenDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testConfirmationToken(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setConfirmationToken($value);
        self::assertNotNull($instance->getConfirmationToken());
        self::assertNotNull($instance->confirmation_token);
        self::assertEquals($value, $instance->getConfirmationToken());
        self::assertEquals($value, $instance->confirmation_token);
    }

    /**
     * Test invalid property "confirmation_token"
     * @dataProvider invalidConfirmationTokenDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidConfirmationToken(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setConfirmationToken($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validConfirmationTokenDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_confirmation_token'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidConfirmationTokenDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_confirmation_token'));
    }
}
