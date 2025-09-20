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

namespace Tests\YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificateApprovedPaymentArticle;

/**
 * ElectronicCertificateApprovedPaymentArticleTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
*/
class ElectronicCertificateApprovedPaymentArticleTest extends AbstractTestCase
{
    protected ElectronicCertificateApprovedPaymentArticle $object;

    /**
    * @return ElectronicCertificateApprovedPaymentArticle
    */
    protected function getTestInstance(): ElectronicCertificateApprovedPaymentArticle
    {
        return new ElectronicCertificateApprovedPaymentArticle();
    }

    /**
    * @return void
    */
    public function testElectronicCertificateApprovedPaymentArticleClassExists(): void
    {
        $this->object = $this->getMockBuilder(ElectronicCertificateApprovedPaymentArticle::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(ElectronicCertificateApprovedPaymentArticle::class));
        $this->assertInstanceOf(ElectronicCertificateApprovedPaymentArticle::class, $this->object);
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
        self::assertLessThanOrEqual(999, is_string($instance->getArticleNumber()) ? mb_strlen($instance->getArticleNumber()) : $instance->getArticleNumber());
        self::assertLessThanOrEqual(999, is_string($instance->article_number) ? mb_strlen($instance->article_number) : $instance->article_number);
        self::assertGreaterThanOrEqual(1, is_string($instance->getArticleNumber()) ? mb_strlen($instance->getArticleNumber()) : $instance->getArticleNumber());
        self::assertGreaterThanOrEqual(1, is_string($instance->article_number) ? mb_strlen($instance->article_number) : $instance->article_number);
        self::assertIsNumeric($instance->getArticleNumber());
        self::assertIsNumeric($instance->article_number);
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
    * Test property "article_code"
    * @dataProvider validArticleCodeDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testArticleCode(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getArticleCode());
        self::assertEmpty($instance->article_code);
        $instance->setArticleCode($value);
        self::assertEquals($value, is_array($value) ? $instance->getArticleCode()->toArray() : $instance->getArticleCode());
        self::assertEquals($value, is_array($value) ? $instance->article_code->toArray() : $instance->article_code);
        if (!empty($value)) {
            self::assertNotNull($instance->getArticleCode());
            self::assertNotNull($instance->article_code);
            self::assertLessThanOrEqual(128, is_string($instance->getArticleCode()) ? mb_strlen($instance->getArticleCode()) : $instance->getArticleCode());
            self::assertLessThanOrEqual(128, is_string($instance->article_code) ? mb_strlen($instance->article_code) : $instance->article_code);
        }
    }

    /**
    * Test invalid property "article_code"
    * @dataProvider invalidArticleCodeDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidArticleCode(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setArticleCode($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validArticleCodeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_article_code'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidArticleCodeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_article_code'));
    }

    /**
    * Test property "certificates"
    * @dataProvider validCertificatesDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testCertificates(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertIsObject($instance->getCertificates());
        self::assertIsObject($instance->certificates);
        self::assertCount(0, $instance->getCertificates());
        self::assertCount(0, $instance->certificates);
        $instance->setCertificates($value);
        self::assertNotNull($instance->getCertificates());
        self::assertNotNull($instance->certificates);
        foreach ($value as $key => $element) {
            if (is_array($element) && !empty($element)) {
                self::assertEquals($element, $instance->getCertificates()[$key]->toArray());
                self::assertEquals($element, $instance->certificates[$key]->toArray());
                self::assertIsArray($instance->getCertificates()[$key]->toArray());
                self::assertIsArray($instance->certificates[$key]->toArray());
            }
            if (is_object($element) && !empty($element)) {
                self::assertEquals($element, $instance->getCertificates()->get($key));
                self::assertIsObject($instance->getCertificates()->get($key));
                self::assertIsObject($instance->certificates->get($key));
                self::assertIsObject($instance->getCertificates());
                self::assertIsObject($instance->certificates);
            }
        }
        self::assertCount(count($value), $instance->getCertificates());
        self::assertCount(count($value), $instance->certificates);
    }

    /**
    * Test invalid property "certificates"
    * @dataProvider invalidCertificatesDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidCertificates(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCertificates($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validCertificatesDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_certificates'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidCertificatesDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_certificates'));
    }
}
