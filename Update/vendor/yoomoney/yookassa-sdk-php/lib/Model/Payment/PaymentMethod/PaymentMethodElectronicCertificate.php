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

namespace YooKassa\Model\Payment\PaymentMethod;

use YooKassa\Common\ListObject;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificateApprovedPaymentArticle;
use YooKassa\Model\Payment\PaymentMethod\ElectronicCertificate\ElectronicCertificatePaymentData;
use YooKassa\Model\Payment\PaymentMethodType;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PaymentMethodElectronicCertificate.
 *
 * Оплата по электронному сертификату.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $type Код способа оплаты.
 * @property BankCard|null $card Данные банковской карты «Мир».
 * @property ElectronicCertificatePaymentData|null $electronic_certificate Данные от ФЭС НСПК для оплаты по электронному сертификату.
 * @property ElectronicCertificatePaymentData|null $electronicCertificate Данные от ФЭС НСПК для оплаты по электронному сертификату.
 * @property ElectronicCertificateApprovedPaymentArticle[]|ListObjectInterface|null $articles Одобренная корзина покупки — список товаров, одобренных к оплате по электронному сертификату.  Присутствует только при [оплате на готовой странице ЮKassa](/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/ready-made-payment-form).
 */
class PaymentMethodElectronicCertificate extends AbstractPaymentMethod
{
    /**
     * @var BankCard|null
     */
    #[Assert\Type(BankCard::class)]
    private ?BankCard $_card = null;

    /**
     * Данные от ФЭС НСПК для оплаты по электронному сертификату.
     *
     * @var ElectronicCertificatePaymentData|null
     */
    #[Assert\Type(ElectronicCertificatePaymentData::class)]
    private ?ElectronicCertificatePaymentData $_electronic_certificate = null;

    /**
     * Одобренная корзина покупки — список товаров, одобренных к оплате по электронному сертификату.  Присутствует только при [оплате на готовой странице ЮKassa](/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/ready-made-payment-form).
     *
     * @var ElectronicCertificateApprovedPaymentArticle[]|ListObjectInterface|null
     */
    #[Assert\AllType(ElectronicCertificateApprovedPaymentArticle::class)]
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
     * @return BankCard|null Данные банковской карты «Мир»
     */
    public function getCard(): ?BankCard
    {
        return $this->_card;
    }

    /**
     * Устанавливает card.
     *
     * @param BankCard|array|null $card Данные банковской карты «Мир»
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
     * @return ElectronicCertificatePaymentData|null Данные от ФЭС НСПК для оплаты по электронному сертификату
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
     * @return ElectronicCertificateApprovedPaymentArticle[]|ListObjectInterface|null Одобренная корзина покупки — список товаров
     */
    public function getArticles(): ?ListObjectInterface
    {
        if ($this->_articles === null) {
            $this->_articles = new ListObject(ElectronicCertificateApprovedPaymentArticle::class);
        }
        return $this->_articles;
    }

    /**
     * Устанавливает articles.
     *
     * @param ListObjectInterface|array|null $articles Одобренная корзина покупки — список товаров, одобренных к оплате по электронному сертификату.  Присутствует только при [оплате на готовой странице ЮKassa](/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/ready-made-payment-form).
     *
     * @return self
     */
    public function setArticles(mixed $articles = null): self
    {
        $this->_articles = $this->validatePropertyValue('_articles', $articles);
        return $this;
    }

}

