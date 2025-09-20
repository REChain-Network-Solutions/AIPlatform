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

namespace YooKassa\Request\Payments\PaymentData;

use YooKassa\Common\ListObject;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificatePaymentData;
use YooKassa\Model\Payment\PaymentMethodType;
use YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PaymentMethodDataElectronicCertificate.
 *
 * Данные для оплаты по электронному сертификату.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $type Код способа оплаты.
 * @property PaymentDataBankCardCard|null $card Данные банковской карты «Мир».
 * @property ElectronicCertificatePaymentData|null $electronic_certificate Данные от ФЭС НСПК для оплаты по электронному сертификату.
 * @property ElectronicCertificatePaymentData|null $electronicCertificate Данные от ФЭС НСПК для оплаты по электронному сертификату.
 * @property ElectronicCertificateArticle[]|ListObjectInterface|null $articles Корзина покупки (в терминах НСПК) — список товаров, которые можно оплатить по сертификату.  Необходимо передавать только при [оплате на готовой странице ЮKassa](/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/ready-made-payment-form).
 */
class PaymentDataElectronicCertificate extends AbstractPaymentData
{
    /**
     * Данные банковской карты «Мир».
     *
     * @var PaymentDataBankCardCard|null
     */
    #[Assert\Type(PaymentDataBankCardCard::class)]
    private ?PaymentDataBankCardCard $_card = null;

    /**
     * Данные от ФЭС НСПК для оплаты по электронному сертификату.
     *
     * @var ElectronicCertificatePaymentData|null
     */
    #[Assert\Type(ElectronicCertificatePaymentData::class)]
    private ?ElectronicCertificatePaymentData $_electronic_certificate = null;

    /**
     * Корзина покупки (в терминах НСПК) — список товаров, которые можно оплатить по сертификату.  Необходимо передавать только при [оплате на готовой странице ЮKassa](/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/ready-made-payment-form).
     *
     * @var ElectronicCertificateArticle[]|ListObjectInterface|null
     */
    #[Assert\AllType(ElectronicCertificateArticle::class)]
    #[Assert\Type(ListObject::class)]
    private ?ListObjectInterface $_articles = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(PaymentMethodType::ELECTRONIC_CERTIFICATE);
    }

    /**
     * Возвращает card.
     *
     * @return PaymentDataBankCardCard|null Данные банковской карты «Мир».
     */
    public function getCard(): ?PaymentDataBankCardCard
    {
        return $this->_card;
    }

    /**
     * Устанавливает card.
     *
     * @param PaymentDataBankCardCard|array|null $card Данные банковской карты «Мир».
     *
     * @return self
     */
    public function setCard(mixed $card = null): self
    {
        $this->_card = $this->validatePropertyValue('_card', $card);
        return $this;
    }

    /**
     * Возвращает electronic_certificate.
     *
     * @return ElectronicCertificatePaymentData|null Данные от ФЭС НСПК для оплаты по электронному сертификату.
     */
    public function getElectronicCertificate(): ?ElectronicCertificatePaymentData
    {
        return $this->_electronic_certificate;
    }

    /**
     * Устанавливает electronic_certificate.
     *
     * @param ElectronicCertificatePaymentData|array|null $electronic_certificate Данные от ФЭС НСПК для оплаты по электронному сертификату.
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
     * @return ElectronicCertificateArticle[]|ListObjectInterface|null Корзина покупки (в терминах НСПК) — список товаров
     */
    public function getArticles(): ?ListObjectInterface
    {
        if ($this->_articles === null) {
            $this->_articles = new ListObject(ElectronicCertificateArticle::class);
        }
        return $this->_articles;
    }

    /**
     * Устанавливает articles.
     *
     * @param ListObjectInterface|array|null $articles Корзина покупки (в терминах НСПК) — список товаров, которые можно оплатить по сертификату.  Необходимо передавать только при [оплате на готовой странице ЮKassa](/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/ready-made-payment-form).
     *
     * @return self
     */
    public function setArticles(mixed $articles = null): self
    {
        $this->_articles = $this->validatePropertyValue('_articles', $articles);
        return $this;
    }

}

