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

namespace YooKassa\Request\Receipts;

use YooKassa\Common\AbstractRequestBuilder;
use YooKassa\Common\AbstractRequestInterface;
use YooKassa\Common\Exceptions\InvalidRequestException;
use YooKassa\Model\Receipt\AdditionalUserProps;
use YooKassa\Model\Receipt\IndustryDetails;
use YooKassa\Model\Receipt\OperationalDetails;
use YooKassa\Model\Receipt\ReceiptCustomerInterface;
use YooKassa\Model\Receipt\ReceiptItemInterface;
use YooKassa\Model\Receipt\ReceiptType;
use YooKassa\Model\Receipt\SettlementInterface;

/**
 * Класс билдера объектов запросов к API на создание чека.
 *
 * @example 02-builder.php 88 57 Пример использования билдера
 */
class CreatePostReceiptRequestBuilder extends AbstractRequestBuilder
{
    /**
     * Собираемый объект запроса.
     *
     * @var CreatePostReceiptRequest|null
     */
    protected ?AbstractRequestInterface $currentObject = null;

    /**
     * Устанавливает валюту в которой будет происходить подтверждение оплаты заказа.
     *
     * @param string $value Валюта в которой подтверждается оплата
     *
     * @return self Инстанс билдера запросов
     */
    public function setCurrency(string $value): self
    {
        foreach ($this->currentObject->getItems() as $item) {
            $item->getPrice()->setCurrency($value);
        }

        return $this;
    }

    /**
     * Устанавливает информацию о пользователе.
     *
     * @param array|ReceiptCustomerInterface $value Информация о плательщике
     *
     * @return self Инстанс билдера запросов
     */
    public function setCustomer(mixed $value): self
    {
        $this->currentObject->setCustomer($value);

        return $this;
    }

    /**
     * Устанавливает список товаров чека.
     *
     * @param array|ReceiptItemInterface[]|null $value Список товаров чека
     *
     * @return self Инстанс билдера запросов
     */
    public function setItems(mixed $value): self
    {
        $this->currentObject->setItems($value);

        return $this;
    }

    /**
     * Добавляет товар в чек.
     *
     * @param array|ReceiptItemInterface|null $value Информация о товаре
     *
     * @return self Инстанс билдера запросов
     */
    public function addItem(mixed $value): self
    {
        $this->currentObject->getItems()->add($value);

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
        $this->currentObject->setTaxSystemCode($value);

        return $this;
    }

    /**
     * Устанавливает дополнительный реквизит пользователя.
     *
     * @param AdditionalUserProps|array|null $value Дополнительный реквизит пользователя
     *
     * @return self Инстанс билдера запросов
     */
    public function setAdditionalUserProps(mixed $value): self
    {
        $this->currentObject->setAdditionalUserProps($value);

        return $this;
    }

    /**
     * Устанавливает отраслевой реквизит чека.
     *
     * @param array|IndustryDetails[]|null $value Отраслевой реквизит чека
     *
     * @return self Инстанс билдера запросов
     */
    public function setReceiptIndustryDetails(mixed $value): self
    {
        $this->currentObject->setReceiptIndustryDetails($value);

        return $this;
    }

    /**
     * Добавляет отраслевой реквизит чека.
     *
     * @param array|IndustryDetails|null $value Отраслевой реквизит чека
     *
     * @return self Инстанс билдера запросов
     */
    public function addReceiptIndustryDetails(mixed $value): self
    {
        $this->currentObject->getReceiptIndustryDetails()->add($value);

        return $this;
    }

    /**
     * Устанавливает операционный реквизит чека.
     *
     * @param array|OperationalDetails|null $value Операционный реквизит чека
     *
     * @return self Инстанс билдера запросов
     */
    public function setReceiptOperationalDetails(mixed $value): self
    {
        $this->currentObject->setReceiptOperationalDetails($value);

        return $this;
    }

    /**
     * Устанавливает тип чека в онлайн-кассе.
     *
     * @param string|null $value Тип чека в онлайн-кассе: приход "payment" или возврат "refund"
     *
     * @return self Инстанс билдера запросов
     */
    public function setType(?string $value): self
    {
        $this->currentObject->setType($value);

        return $this;
    }

    /**
     * Устанавливает признак отложенной отправки чека.
     *
     * @param bool $value Признак отложенной отправки чека
     *
     * @return self Инстанс билдера запросов
     */
    public function setSend(bool $value): self
    {
        $this->currentObject->setSend($value);

        return $this;
    }

    /**
     * Устанавливает идентификатор магазина, от имени которого нужно отправить чек.
     * Выдается ЮKassa, отображается в разделе Продавцы личного кабинета (столбец shopId).
     * Необходимо передавать, если вы используете решение ЮKassa для платформ.
     *
     * @param string|null $value Идентификатор магазина, от имени которого нужно отправить чек
     *
     * @return self Инстанс билдера запросов
     */
    public function setOnBehalfOf(?string $value): self
    {
        $this->currentObject->setOnBehalfOf($value);

        return $this;
    }

    /**
     * Устанавливает массив оплат, обеспечивающих выдачу товара.
     *
     * @param array|SettlementInterface[]|null $value Массив оплат, обеспечивающих выдачу товара
     *
     * @return self Инстанс билдера запросов
     */
    public function setSettlements(mixed $value): self
    {
        $this->currentObject->setSettlements($value);

        return $this;
    }

    /**
     * Добавляет оплату в перечень совершенных расчетов.
     *
     * @param array|SettlementInterface|null $value Информация об оплате
     *
     * @return self Инстанс билдера запросов
     */
    public function addSettlement(mixed $value): self
    {
        $this->currentObject->getSettlements()->add($value);

        return $this;
    }

    /**
     * Устанавливает Id объекта чека.
     *
     * @param string $value Id объекта чека
     * @param null|string $type Тип объекта чека
     *
     * @return self Инстанс билдера запросов
     */
    public function setObjectId(string $value, ?string $type = null): self
    {
        $this->currentObject->setObjectId($value);
        if (!empty($type)) {
            $this->currentObject->setObjectType($type);
        }

        return $this;
    }

    /**
     * Устанавливает тип объекта чека.
     *
     * @param string|null $value Тип объекта чека
     *
     * @return self Инстанс билдера запросов
     */
    public function setObjectType(?string $value): self
    {
        $this->currentObject->setObjectType($value);

        return $this;
    }

    /**
     * Строит и возвращает объект запроса для отправки в API ЮKassa.
     *
     * @param null|array $options Массив параметров для установки в объект запроса
     *
     * @return CreatePostReceiptRequest Инстанс объекта запроса
     *
     * @throws InvalidRequestException Выбрасывается если собрать объект запроса не удалось
     */
    public function build(?array $options = null): AbstractRequestInterface
    {
        if (!empty($options)) {
            $this->setOptions($options);

            if (!empty($options['payment_id'])) {
                $this->setObjectId($options['payment_id']);
                $this->setObjectType(ReceiptType::PAYMENT);
            } elseif (!empty($options['refund_id'])) {
                $this->setObjectId($options['refund_id']);
                $this->setObjectType(ReceiptType::REFUND);
            }
        }

        return parent::build($options);
    }

    /**
     * Инициализирует объект запроса, который в дальнейшем будет собираться билдером
     *
     * @return CreatePostReceiptRequest Инстанс собираемого объекта запроса к API
     */
    protected function initCurrentObject(): CreatePostReceiptRequest
    {
        return new CreatePostReceiptRequest();
    }
}
