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

namespace Tests\YooKassa\Model\Refund\RefundMethod\ElectronicCertificate;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use YooKassa\Model\Refund\RefundMethod\ElectronicCertificate\ElectronicCertificateRefundArticle;

/**
 * ElectronicCertificateRefundArticleTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
*/
class ElectronicCertificateRefundArticleTest extends AbstractTestCase
{
    protected ElectronicCertificateRefundArticle $object;

    /**
    * @return ElectronicCertificateRefundArticle
    */
    protected function getTestInstance(): ElectronicCertificateRefundArticle
    {
        return new ElectronicCertificateRefundArticle();
    }

    /**
    * @return void
    */
    public function testElectronicCertificateRefundArticleClassExists(): void
    {
        $this->object = $this->getMockBuilder(ElectronicCertificateRefundArticle::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(ElectronicCertificateRefundArticle::class));
        $this->assertInstanceOf(ElectronicCertificateRefundArticle::class, $this->object);
    }

    /**
    * Test property "article_number"
    * @dataProvider validArticleNumberDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testArticleNumber(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setArticleNumber($value);
        self::assertNotNull($instance->getArticleNumber());
        self::assertNotNull($instance->article_number);
        self::assertEquals($value, is_array($value) ? $instance->getArticleNumber()->toArray() : $instance->getArticleNumber());
        self::assertEquals($value, is_array($value) ? $instance->article_number->toArray() : $instance->article_number);
    }

    /**
    * Test invalid property "article_number"
    * @dataProvider invalidArticleNumberDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidArticleNumber(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setArticleNumber($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validArticleNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_article_number'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidArticleNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_article_number'));
    }

    /**
    * Test property "payment_article_number"
    * @dataProvider validPaymentArticleNumberDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testPaymentArticleNumber(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setPaymentArticleNumber($value);
        self::assertNotNull($instance->getPaymentArticleNumber());
        self::assertNotNull($instance->payment_article_number);
        self::assertEquals($value, is_array($value) ? $instance->getPaymentArticleNumber()->toArray() : $instance->getPaymentArticleNumber());
        self::assertEquals($value, is_array($value) ? $instance->payment_article_number->toArray() : $instance->payment_article_number);
    }

    /**
    * Test invalid property "payment_article_number"
    * @dataProvider invalidPaymentArticleNumberDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidPaymentArticleNumber(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPaymentArticleNumber($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validPaymentArticleNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_article_number'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidPaymentArticleNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_payment_article_number'));
    }

    /**
    * Test property "tru_code"
    * @dataProvider validTruCodeDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testTruCode(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setTruCode($value);
        self::assertNotNull($instance->getTruCode());
        self::assertNotNull($instance->tru_code);
        self::assertEquals($value, is_array($value) ? $instance->getTruCode()->toArray() : $instance->getTruCode());
        self::assertEquals($value, is_array($value) ? $instance->tru_code->toArray() : $instance->tru_code);
        self::assertLessThanOrEqual(30, is_string($instance->getTruCode()) ? mb_strlen($instance->getTruCode()) : $instance->getTruCode());
        self::assertLessThanOrEqual(30, is_string($instance->tru_code) ? mb_strlen($instance->tru_code) : $instance->tru_code);
        self::assertGreaterThanOrEqual(30, is_string($instance->getTruCode()) ? mb_strlen($instance->getTruCode()) : $instance->getTruCode());
        self::assertGreaterThanOrEqual(30, is_string($instance->tru_code) ? mb_strlen($instance->tru_code) : $instance->tru_code);
    }

    /**
    * Test invalid property "tru_code"
    * @dataProvider invalidTruCodeDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidTruCode(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setTruCode($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validTruCodeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_tru_code'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidTruCodeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_tru_code'));
    }

    /**
    * Test property "quantity"
    * @dataProvider validQuantityDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testQuantity(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setQuantity($value);
        self::assertNotNull($instance->getQuantity());
        self::assertNotNull($instance->quantity);
        self::assertEquals($value, is_array($value) ? $instance->getQuantity()->toArray() : $instance->getQuantity());
        self::assertEquals($value, is_array($value) ? $instance->quantity->toArray() : $instance->quantity);
        self::assertGreaterThanOrEqual(1, is_string($instance->getQuantity()) ? mb_strlen($instance->getQuantity()) : $instance->getQuantity());
        self::assertGreaterThanOrEqual(1, is_string($instance->quantity) ? mb_strlen($instance->quantity) : $instance->quantity);
        self::assertIsNumeric($instance->getQuantity());
        self::assertIsNumeric($instance->quantity);
    }

    /**
    * Test invalid property "quantity"
    * @dataProvider invalidQuantityDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidQuantity(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setQuantity($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validQuantityDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_quantity'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidQuantityDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_quantity'));
    }
}
