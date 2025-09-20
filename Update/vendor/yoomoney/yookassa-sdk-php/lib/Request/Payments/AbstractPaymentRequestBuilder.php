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

namespace YooKassa\Request\Payments;

use YooKassa\Common\AbstractRequestBuilder;
use YooKassa\Common\AbstractRequestInterface;
use YooKassa\Common\Exceptions\InvalidPropertyValueException;
use YooKassa\Common\Exceptions\InvalidPropertyValueTypeException;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\AmountInterface;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Model\Receipt\Receipt;
use YooKassa\Model\Receipt\ReceiptCustomer;
use YooKassa\Model\Receipt\ReceiptInterface;
use YooKassa\Model\Receipt\ReceiptItem;
use YooKassa\Model\Receipt\ReceiptItemAmount;
use YooKassa\Model\Receipt\ReceiptItemInterface;

/**
 * Класс, представляющий модель AbstractPaymentRequestBuilder.
 *
 * Базовый класс билдера объекта платежного запроса, передаваемого в методы клиента API.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
abstract class AbstractPaymentRequestBuilder extends AbstractRequestBuilder
{
    /**
     * Сумма
     * @var MonetaryAmount|null
     */
    protected ?MonetaryAmount $amount = null;

    /**
     * Объект с информацией о чеке
     * @var Receipt|null
     */
    protected ?Receipt $receipt = null;

    /**
     * {@inheritDoc}
     */
    public function build(?array $options = null): AbstractRequestInterface
    {
        return parent::build($options);
    }

    /**
     * Устанавливает сумму.
     *
     * @param AmountInterface|array|numeric $value Сумма оплаты
     * @param string|null $currency Валюта
     *
     * @return self Инстанс билдера запросов
     */
    public function setAmount(mixed $value, ?string $currency = null): self
    {
        if (null === $value || '' === $value) {
            $this->amount = new MonetaryAmount();
        } elseif ($value instanceof AmountInterface) {
            $this->amount->setValue($value->getValue());
            $this->amount->setCurrency($value->getCurrency());
        } elseif (is_array($value)) {
            $this->amount->fromArray($value);
        } else {
            $this->amount->setValue($value);
            if (!empty($currency)) {
                $this->amount->setCurrency($currency);
            }
        }

        return $this;
    }

    /**
     * Устанавливает трансферы.
     *
     * @param array|TransferDataInterface[]|ListObjectInterface $value Массив трансферов
     *
     * @return self Инстанс билдера запросов
     */
    public function setTransfers(mixed $value): self
    {
        $this->currentObject->setTransfers($value);

        return $this;
    }

    /**
     * Добавляет трансфер.
     *
     * @param array|TransferDataInterface $value Трансфер
     *
     * @return self Инстанс билдера запросов
     */
    public function addTransfer(mixed $value): self
    {
        $this->currentObject->getTransfers()->add($value);

        return $this;
    }

    /**
     * Устанавливает валюту в которой будет происходить подтверждение оплаты заказа.
     *
     * @param string $value Валюта в которой подтверждается оплата
     *
     * @return self Инстанс билдера запросов
     */
    public function setCurrency(string $value): self
    {
        $this->amount->setCurrency($value);
        foreach ($this->receipt->getItems() as $item) {
            $item->getPrice()->setCurrency($value);
        }

        return $this;
    }

    /**
     * Устанавливает чек.
     *
     * @param array|ReceiptInterface $value Инстанс чека или ассоциативный массив с данными чека
     *
     * @return self Инстанс билдера запросов
     *
     * @throws InvalidPropertyValueTypeException Генерируется если было передано значение невалидного типа
     */
    public function setReceipt(mixed $value): self
    {
        if (is_array($value)) {
            $this->receipt->fromArray($value);
        } elseif ($value instanceof ReceiptInterface) {
            $this->receipt = clone $value;
        } else {
            throw new InvalidPropertyValueTypeException('Invalid receipt value type', 0, 'receipt', $value);
        }

        return $this;
    }

    /**
     * Устанавливает список товаров для создания чека.
     *
     * @param array $value Массив товаров в заказе
     *
     * @return self Инстанс билдера запросов
     *
     * @throws InvalidPropertyValueException Выбрасывается если хотя бы один из товаров имеет неверную структуру
     */
    public function setReceiptItems(array $value = []): self
    {
        $this->receipt->removeItems();
        foreach ($value as $item) {
            if ($item instanceof ReceiptItemInterface) {
                $this->receipt->addItem($item);
            } else {
                $this->receipt->addItem(new ReceiptItem($item));
            }
        }

        return $this;
    }

    /**
     * Добавляет в чек товар
     *
     * @param string $title Название или описание товара
     * @param string $price Цена товара в валюте, заданной в заказе
     * @param float $quantity Количество товара
     * @param int $vatCode Ставка НДС
     * @param null|string $paymentSubject значение перечисления PaymentSubject
     * @param null|string $paymentMode значение перечисления PaymentMode
     * @param null|mixed $productCode
     * @param null|mixed $countryOfOriginCode
     * @param null|mixed $customsDeclarationNumber
     * @param null|mixed $excise
     *
     * @return self Инстанс билдера запросов
     *
     * @see PaymentSubject
     * @see PaymentMode
     */
    public function addReceiptItem(string $title, string $price, float $quantity, int $vatCode, ?string $paymentMode = null, ?string $paymentSubject = null, $productCode = null, $countryOfOriginCode = null, $customsDeclarationNumber = null, $excise = null): self
    {
        $item = new ReceiptItem();
        $item->setDescription($title);
        $item->setQuantity($quantity);
        $item->setVatCode($vatCode);
        $item->setPrice(new ReceiptItemAmount($price, $this->amount->getCurrency()));
        $item->setPaymentSubject($paymentSubject);
        $item->setPaymentMode($paymentMode);
        $item->setProductCode($productCode);
        $item->setCountryOfOriginCode($countryOfOriginCode);
        $item->setCustomsDeclarationNumber($customsDeclarationNumber);
        $item->setExcise($excise);
        $this->receipt->addItem($item);

        return $this;
    }

    /**
     * Добавляет в чек доставку товара.
     *
     * @param string $title Название доставки в чеке
     * @param string $price Стоимость доставки
     * @param int $vatCode Ставка НДС
     * @param null|string $paymentSubject значение перечисления PaymentSubject
     * @param null|string $paymentMode значение перечисления PaymentMode
     *
     * @return self Инстанс билдера запросов
     *
     * @see PaymentSubject
     * @see PaymentMode
     */
    public function addReceiptShipping(string $title, string $price, int $vatCode, ?string $paymentMode = null, ?string $paymentSubject = null): self
    {
        $item = new ReceiptItem();
        $item->setDescription($title);
        $item->setQuantity(1);
        $item->setVatCode($vatCode);
        $item->setIsShipping(true);
        $item->setPrice(new ReceiptItemAmount($price, $this->amount->getCurrency()));
        $item->setPaymentMode($paymentMode);
        $item->setPaymentSubject($paymentSubject);
        $this->receipt->addItem($item);

        return $this;
    }

    /**
     * Устанавливает адрес электронной почты получателя чека.
     *
     * @param string|null $value Email получателя чека
     *
     * @return self Инстанс билдера запросов
     */
    public function setReceiptEmail(?string $value): self
    {
        if (!$this->receipt->getCustomer()) {
            $this->receipt->setCustomer(new ReceiptCustomer());
        }
        $this->receipt->getCustomer()->setEmail($value);

        return $this;
    }

    /**
     * Устанавливает телефон получателя чека.
     *
     * @param string|null $value Телефон получателя чека
     *
     * @return self Инстанс билдера запросов
     */
    public function setReceiptPhone(?string $value): self
    {
        if (!$this->receipt->getCustomer()) {
            $this->receipt->setCustomer(new ReceiptCustomer());
        }
        $this->receipt->getCustomer()->setPhone($value);

        return $this;
    }

    /**
     * Устанавливает код системы налогообложения.
     *
     * @param int|null $value Код системы налогообложения. Число 1-6.
     *
     * @return self Инстанс билдера запросов
     */
    public function setTaxSystemCode(?int $value): self
    {
        $this->receipt->setTaxSystemCode($value);

        return $this;
    }

    /**
     * Инициализирует пустой запрос
     *
     * @return AbstractRequestInterface|null
     */
    protected function initCurrentObject(): ?AbstractRequestInterface
    {
        $this->amount = new MonetaryAmount();
        $this->receipt = new Receipt();

        return new AbstractPaymentRequest();
    }
}
