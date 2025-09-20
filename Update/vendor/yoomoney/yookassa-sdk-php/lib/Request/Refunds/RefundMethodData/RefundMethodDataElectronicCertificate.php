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

namespace YooKassa\Request\Refunds\RefundMethodData;

use YooKassa\Common\ListObject;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\Refund\RefundMethod\ElectronicCertificate\ElectronicCertificateRefundArticle;
use YooKassa\Model\Refund\RefundMethod\ElectronicCertificate\ElectronicCertificateRefundData;
use YooKassa\Model\Refund\RefundMethodType;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель ElectronicCertificateRefundMethod.
 *
 * Возврат платежа по электронному сертификату.
 *
 * @category Class
 * @package  YooKassa\Modela
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $type Код способа оплаты.
 * @property ElectronicCertificateRefundData $electronic_certificate Данные от ФЭС НСПК для возврата на электронный сертификат. Необходимо передавать только при [оплате со сбором данных на вашей стороне](https://yookassa.ru/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/merchant-payment-form).
 * @property ElectronicCertificateRefundData $electronicCertificate Данные от ФЭС НСПК для возврата на электронный сертификат. Необходимо передавать только при [оплате со сбором данных на вашей стороне](https://yookassa.ru/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/merchant-payment-form).
 * @property ElectronicCertificateRefundArticle[]|ListObjectInterface $articles Корзина возврата (в терминах НСПК) — список возвращаемых товаров, для оплаты которых использовался электронный сертификат. Данные должны соответствовать товарам из одобренной корзины покупки (`articles` в [объекте платежа](#payment_object)).  Необходимо передавать только при [оплате на готовой странице ЮKassa](/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/ready-made-payment-form).
 */
class RefundMethodDataElectronicCertificate extends AbstractRefundMethodData
{
    /**
     * Данные от ФЭС НСПК для возврата на электронный сертификат.
     * Необходимо передавать только при [оплате со сбором данных на вашей стороне](https://yookassa.ru/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/merchant-payment-form).
     *
     * @var ElectronicCertificateRefundData|null
     */
    #[Assert\Type(ElectronicCertificateRefundData::class)]
    private ?ElectronicCertificateRefundData $_electronic_certificate = null;

    /**
     * Корзина возврата (в терминах НСПК) — список возвращаемых товаров, для оплаты которых использовался электронный сертификат. Данные должны соответствовать товарам из одобренной корзины покупки (`articles` в [объекте платежа](#payment_object)).  Необходимо передавать только при [оплате на готовой странице ЮKassa](/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/ready-made-payment-form).
     *
     * @var ElectronicCertificateRefundArticle[]|ListObjectInterface|null
     */
    #[Assert\AllType(ElectronicCertificateRefundArticle::class)]
    #[Assert\Type(ListObject::class)]
    private ?ListObjectInterface $_articles = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(RefundMethodType::ELECTRONIC_CERTIFICATE);
    }

    /**
     * Возвращает electronic_certificate.
     *
     * @return ElectronicCertificateRefundData|null Данные от ФЭС НСПК для возврата на электронный сертификат.
     */
    public function getElectronicCertificate(): ?ElectronicCertificateRefundData
    {
        return $this->_electronic_certificate;
    }

    /**
     * Устанавливает electronic_certificate.
     *
     * @param ElectronicCertificateRefundData|array|null $electronic_certificate Данные от ФЭС НСПК для возврата на электронный сертификат.
     *
     * @return self
     */
    public function setElectronicCertificate(mixed $electronic_certificate = null): self
    {
        $this->_electronic_certificate = $this->validatePropertyValue('_electronic_certificate', $electronic_certificate);
        return $this;
    }

    /**
     * Возвращает articles.
     *
     * @return ElectronicCertificateRefundArticle[]|ListObjectInterface|null Корзина возврата (в терминах НСПК) — список возвращаемых товаров
     */
    public function getArticles(): ?ListObjectInterface
    {
        if ($this->_articles === null) {
            $this->_articles = new ListObject(ElectronicCertificateRefundArticle::class);
        }
        return $this->_articles;
    }

    /**
     * Устанавливает articles.
     *
     * @param ListObjectInterface|array|null $articles Корзина возврата (в терминах НСПК) — список возвращаемых товаров, для оплаты которых использовался электронный сертификат. Данные должны соответствовать товарам из одобренной корзины покупки (`articles` в [объекте платежа](#payment_object)).  Необходимо передавать только при [оплате на готовой странице ЮKassa](/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/ready-made-payment-form).
     *
     * @return self
     */
    public function setArticles(mixed $articles = null): self
    {
        $this->_articles = $this->validatePropertyValue('_articles', $articles);
        return $this;
    }

}

