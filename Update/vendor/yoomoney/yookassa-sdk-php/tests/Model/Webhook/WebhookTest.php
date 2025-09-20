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

namespace Tests\YooKassa\Model\Webhook;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Webhook\Webhook;

/**
 * WebhookTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class WebhookTest extends AbstractTestCase
{
    protected Webhook $object;

    /**
     * @return Webhook
     */
    protected function getTestInstance(): Webhook
    {
        return new Webhook();
    }

    /**
     * @return void
     */
    public function testWebhookClassExists(): void
    {
        $this->object = $this->getMockBuilder(Webhook::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(Webhook::class));
        $this->assertInstanceOf(Webhook::class, $this->object);
    }

    /**
     * Test property "id"
     * @dataProvider validIdDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setId($value);
        self::assertNotNull($instance->getId());
        self::assertNotNull($instance->id);
        self::assertEquals($value, is_array($value) ? $instance->getId()->toArray() : $instance->getId());
        self::assertEquals($value, is_array($value) ? $instance->id->toArray() : $instance->id);
    }

    /**
     * Test invalid property "id"
     * @dataProvider invalidIdDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setId($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_id'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_id'));
    }

    /**
     * Test property "event"
     * @dataProvider validEventDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testEvent(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setEvent($value);
        self::assertNotNull($instance->getEvent());
        self::assertNotNull($instance->event);
        self::assertEquals($value, is_array($value) ? $instance->getEvent()->toArray() : $instance->getEvent());
        self::assertEquals($value, is_array($value) ? $instance->event->toArray() : $instance->event);
        self::assertContains($instance->getEvent(), ['payment.waiting_for_capture', 'payment.succeeded', 'payment.canceled', 'refund.succeeded', 'deal.closed', 'payout.canceled', 'payout.succeeded']);
        self::assertContains($instance->event, ['payment.waiting_for_capture', 'payment.succeeded', 'payment.canceled', 'refund.succeeded', 'deal.closed', 'payout.canceled', 'payout.succeeded']);
    }

    /**
     * Test invalid property "event"
     * @dataProvider invalidEventDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidEvent(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setEvent($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validEventDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_event'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidEventDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_event'));
    }

    /**
     * Test property "url"
     * @dataProvider validUrlDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testUrl(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setUrl($value);
        self::assertNotNull($instance->getUrl());
        self::assertNotNull($instance->url);
        self::assertEquals($value, is_array($value) ? $instance->getUrl()->toArray() : $instance->getUrl());
        self::assertEquals($value, is_array($value) ? $instance->url->toArray() : $instance->url);
    }

    /**
     * Test invalid property "url"
     * @dataProvider invalidUrlDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidUrl(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setUrl($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validUrlDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_url'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidUrlDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_url'));
    }
}
